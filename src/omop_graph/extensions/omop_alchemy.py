# Extension to omop-alchemy package
import sqlalchemy as sa
import sqlalchemy.orm as so
from orm_loader.helpers import Base
from omop_alchemy.cdm.base import ReferenceTable, cdm_table, CDMTableBase

import functools
from enum import Enum
from dataclasses import dataclass

class ClassIDEnum(Enum):
    HIERARCHY = "Hierarchy"
    IDENTITY = "Identity"
    COMPOSITION = "Composition"
    ASSOCIATION = "Association"
    ATTRIBUTE = "Attribute"

@cdm_table
class RelationshipClass(ReferenceTable, CDMTableBase, Base):
    """
    Extensions table: Defines the semantic categories (Parent) 
    and entity types (Child).
    """
    __tablename__ = "relationship_class"
    class_id: so.Mapped[ClassIDEnum] = so.mapped_column(
        sa.Enum(
            ClassIDEnum,
            values_callable=lambda obj: [e.value for e in obj]  # Use the value of the enum for storage
        ), 
        primary_key=True
    )
    subclass_id: so.Mapped[str] = so.mapped_column(sa.String(20), primary_key=True)
    description: so.Mapped[str] = so.mapped_column(sa.String(80), nullable=False)
    semantics: so.Mapped[str] = so.mapped_column(sa.String(40), nullable=False)
    inference: so.Mapped[str] = so.mapped_column(sa.String(40), nullable=False)

@cdm_table
class RelationshipMapping(ReferenceTable, CDMTableBase, Base):
    """
    Extensions table: Maps standard OMOP relationship_ids to 
    their parent (class_id - one of ClassIDEnum) and more fine-grained subclasses  .
    """
    __tablename__ = "relationship_mapping"

    relationship_id: so.Mapped[str] = so.mapped_column(
        sa.ForeignKey("relationship.relationship_id"), primary_key=True
    )
    class_id: so.Mapped[ClassIDEnum] = so.mapped_column(
        sa.Enum(
            ClassIDEnum,
            values_callable=lambda obj: [e.value for e in obj]  # Use the value of the enum for storage
        ), primary_key=True
    )
    subclass_id: so.Mapped[str] = so.mapped_column(
        sa.String(20), primary_key=True
    )

    # Define the Composite Foreign Key in __table_args__
    __table_args__ = (
        sa.ForeignKeyConstraint(
            ["class_id", "subclass_id"],
            ["relationship_class.class_id", "relationship_class.subclass_id"],
            name="fk_rel_mapping_to_rel_class"
        ),
    )

@dataclass(frozen=True, slots=True)
class RelationshipMappingElement:
    relationship_id: str
    class_id: ClassIDEnum
    subclass_id: str

    @classmethod
    def from_relationship_mapping_entry(cls, entry) -> "RelationshipMappingElement":
        return cls(
            relationship_id=entry.relationship_id,
            class_id=ClassIDEnum(entry.class_id),
            subclass_id=entry.subclass_id
        )



class RelationshipCache:
    """Cache for the RelationshipMapping table in-memory.
    Allows quicker access."""
    _mapping: dict[str, RelationshipMappingElement] = {}
    _is_initialized: bool = False

    @classmethod
    def load(cls, session: so.Session):
        """Loads the entire PredicateMapping table into memory."""
        if cls._is_initialized:
            return
        
        results = session.query(RelationshipMapping).all()
        cls._mapping = {row.relationship_id: RelationshipMappingElement.from_relationship_mapping_entry(row) for row in results}
        cls._is_initialized = True

    @classmethod
    def get(cls, source_concept_id: str) -> RelationshipMappingElement:
        """Retrieves a mapped concept, strictly from memory."""
        if cls._mapping is None:
            raise RuntimeError(
                "PredicateCache was accessed before being initialized. "
                "Ensure PredicateCache.load(session) is called at application startup."
            )
        item = cls._mapping.get(source_concept_id, None)
        if item is None:
            raise AttributeError(f"`{source_concept_id}` not in mapping.")
        return item
    
def validate_mapping_table(func_to_decorate):
    @functools.wraps(func_to_decorate)
    def wrapper(self, *args, **kwargs):
        try:
            factory = self.session_factory
        except AttributeError:
            raise AttributeError(
                "Decorator requires 'self.session_factory' to exist on the class instance."
            )

        engine = factory.kw.get("bind")
        if engine and not sa.inspect(engine).has_table(RelationshipMapping.__tablename__):
            raise RuntimeError("Database table for relationship mapping is missing. This is unexpected.")

        with factory() as session:
            exists = session.scalar(
                sa.select(sa.func.count()).select_from(RelationshipMapping)
            )
            if not exists:
                raise RuntimeError(f"Table '{RelationshipMapping.__tablename__}' has no entries. Did you ingest the new classification using the cli with `omop-graph relationship_classification *args`?")

        return func_to_decorate(self, *args, **kwargs)
    
    return wrapper