from __future__ import annotations

from datetime import date
from typing import cast

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from orm_loader.helpers import Base
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_class import Concept_Class
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.domain import Domain
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary

from omop_graph.extensions.omop_alchemy import (
    ClassIDEnum,
    RelationshipCache,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph

PARENT_CANCER_ID = 443392
CONCEPT_META_ID = 0
LANGUAGE_CONCEPT_ID = 1


@pytest.fixture(scope="module")
def mock_cdm_engine() -> sa.Engine:
    engine = sa.create_engine("sqlite+pysqlite:///:memory:", future=True)
    tables = cast(
        list[sa.Table],
        [
            Concept.__table__,
            Domain.__table__,
            Vocabulary.__table__,
            Concept_Class.__table__,
            Relationship.__table__,
            Concept_Ancestor.__table__,
            Concept_Relationship.__table__,
            Concept_Synonym.__table__,
            RelationshipClass.__table__,
            RelationshipMapping.__table__,
        ],
    )

    Base.metadata.create_all(engine, tables=tables)

    session_local = sessionmaker(bind=engine, future=True)
    with session_local() as session:
        seed_mock_cdm(session)

    return engine


@pytest.fixture()
def mock_cdm_kg(
    mock_cdm_engine: sa.Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> KnowledgeGraph:
    # Ensure cache does not leak between tests.
    RelationshipCache._mapping = {}
    RelationshipCache._is_initialized = False

    # Grounding tests here focus on SQL + resolver + path pipeline.
    monkeypatch.setattr("omop_graph.reasoning.grounding.get_embedding_writer_interface", lambda _kg: None)

    return KnowledgeGraph(cdm_engine=mock_cdm_engine)


def seed_mock_cdm(session: Session) -> None:
    today = date(2020, 1, 1)

    session.add_all(
        [
            Concept(
                concept_id=CONCEPT_META_ID,
                concept_name="Meta concept",
                domain_id="Metadata",
                vocabulary_id="OMOP",
                concept_class_id="Metadata",
                standard_concept="S",
                concept_code="META",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
            Concept(
                concept_id=LANGUAGE_CONCEPT_ID,
                concept_name="English",
                domain_id="Metadata",
                vocabulary_id="OMOP",
                concept_class_id="Metadata",
                standard_concept="S",
                concept_code="EN",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
            Concept(
                concept_id=PARENT_CANCER_ID,
                concept_name="Malignant neoplastic disease",
                domain_id="Condition",
                vocabulary_id="SNOMED",
                concept_class_id="Clinical Finding",
                standard_concept="S",
                concept_code="CANCER_PARENT",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
            Concept(
                concept_id=196653,
                concept_name="Malignant tumor of kidney",
                domain_id="Condition",
                vocabulary_id="SNOMED",
                concept_class_id="Clinical Finding",
                standard_concept="S",
                concept_code="KIDNEY_CA",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
            Concept(
                concept_id=4038835,
                concept_name="Hodgkin's disease (clinical)",
                domain_id="Condition",
                vocabulary_id="SNOMED",
                concept_class_id="Clinical Finding",
                standard_concept="S",
                concept_code="HODGKIN",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
            Concept(
                concept_id=4181351,
                concept_name="Malignant neoplasm of ovary",
                domain_id="Condition",
                vocabulary_id="SNOMED",
                concept_class_id="Clinical Finding",
                standard_concept="S",
                concept_code="OVARY_CA",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
            Concept(
                concept_id=140352,
                concept_name="Acute myeloid leukemia, disease",
                domain_id="Condition",
                vocabulary_id="SNOMED",
                concept_class_id="Clinical Finding",
                standard_concept="S",
                concept_code="AML",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
            Concept(
                concept_id=138994,
                concept_name="Myelodysplastic syndrome (clinical)",
                domain_id="Condition",
                vocabulary_id="SNOMED",
                concept_class_id="Clinical Finding",
                standard_concept="S",
                concept_code="MDS",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
            Concept(
                concept_id=900001,
                concept_name="Kidney carcinoma term",
                domain_id="Condition",
                vocabulary_id="SNOMED",
                concept_class_id="Clinical Finding",
                standard_concept=None,
                concept_code="KIDNEY_NON_STD",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
            Concept(
                concept_id=900999,
                concept_name="Mass of kidney",
                domain_id="Condition",
                vocabulary_id="SNOMED",
                concept_class_id="Clinical Finding",
                standard_concept="S",
                concept_code="DISTRACTOR",
                valid_start_date=today,
                valid_end_date=date(2099, 12, 31),
                invalid_reason=None,
            ),
        ]
    )

    session.add_all(
        [
            Domain(domain_id="Metadata", domain_name="Metadata", domain_concept_id=CONCEPT_META_ID),
            Domain(domain_id="Condition", domain_name="Condition", domain_concept_id=CONCEPT_META_ID),
            Vocabulary(
                vocabulary_id="OMOP",
                vocabulary_name="OMOP",
                vocabulary_reference="local",
                vocabulary_version="test",
                vocabulary_concept_id=CONCEPT_META_ID,
            ),
            Vocabulary(
                vocabulary_id="SNOMED",
                vocabulary_name="SNOMED",
                vocabulary_reference="local",
                vocabulary_version="test",
                vocabulary_concept_id=CONCEPT_META_ID,
            ),
            Concept_Class(
                concept_class_id="Metadata",
                concept_class_name="Metadata",
                concept_class_concept_id=CONCEPT_META_ID,
            ),
            Concept_Class(
                concept_class_id="Clinical Finding",
                concept_class_name="Clinical Finding",
                concept_class_concept_id=CONCEPT_META_ID,
            ),
        ]
    )

    session.add_all(
        [
            Relationship(
                relationship_id="maps to",
                relationship_name="Maps to",
                is_hierarchical="0",
                defines_ancestry="0",
                reverse_relationship_id="mapped from",
                relationship_concept_id=CONCEPT_META_ID,
            ),
            Relationship(
                relationship_id="mapped from",
                relationship_name="Mapped from",
                is_hierarchical="0",
                defines_ancestry="0",
                reverse_relationship_id="maps to",
                relationship_concept_id=CONCEPT_META_ID,
            ),
        ]
    )

    session.add_all(
        [
            RelationshipClass(
                class_id=ClassIDEnum.IDENTITY,
                subclass_id="mapping",
                description="Identity mapping",
                semantics="identity",
                inference="none",
            ),
            RelationshipMapping(
                relationship_id="maps to",
                class_id=ClassIDEnum.IDENTITY,
                subclass_id="mapping",
            ),
            RelationshipMapping(
                relationship_id="mapped from",
                class_id=ClassIDEnum.IDENTITY,
                subclass_id="mapping",
            ),
        ]
    )

    session.add_all(
        [
            Concept_Ancestor(
                ancestor_concept_id=PARENT_CANCER_ID,
                descendant_concept_id=196653,
                min_levels_of_separation=2,
                max_levels_of_separation=2,
            ),
            Concept_Ancestor(
                ancestor_concept_id=PARENT_CANCER_ID,
                descendant_concept_id=4038835,
                min_levels_of_separation=2,
                max_levels_of_separation=2,
            ),
            Concept_Ancestor(
                ancestor_concept_id=PARENT_CANCER_ID,
                descendant_concept_id=4181351,
                min_levels_of_separation=2,
                max_levels_of_separation=2,
            ),
            Concept_Ancestor(
                ancestor_concept_id=PARENT_CANCER_ID,
                descendant_concept_id=140352,
                min_levels_of_separation=2,
                max_levels_of_separation=2,
            ),
            Concept_Ancestor(
                ancestor_concept_id=PARENT_CANCER_ID,
                descendant_concept_id=138994,
                min_levels_of_separation=2,
                max_levels_of_separation=2,
            ),
            Concept_Ancestor(
                ancestor_concept_id=PARENT_CANCER_ID,
                descendant_concept_id=900999,
                min_levels_of_separation=2,
                max_levels_of_separation=2,
            ),
        ]
    )

    session.add_all(
        [
            Concept_Synonym(
                concept_id=196653,
                concept_synonym_name="Kidney cancer",
                language_concept_id=LANGUAGE_CONCEPT_ID,
            ),
            Concept_Synonym(
                concept_id=138994,
                concept_synonym_name="Myelodysplasia",
                language_concept_id=LANGUAGE_CONCEPT_ID,
            ),
            Concept_Synonym(
                concept_id=140352,
                concept_synonym_name="Acute myeloid leukaemia",
                language_concept_id=LANGUAGE_CONCEPT_ID,
            ),
        ]
    )

    session.add(
        Concept_Relationship(
            concept_id_1=900001,
            relationship_id="maps to",
            concept_id_2=196653,
            valid_start_date=today,
            valid_end_date=date(2099, 12, 31),
            invalid_reason=None,
        )
    )

    session.commit()
