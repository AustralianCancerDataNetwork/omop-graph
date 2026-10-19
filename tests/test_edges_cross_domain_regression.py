"""Cross-domain attribute edges are visible by default.

Almost every SNOMED procedure attribute crosses domains: ``Has method`` ->
Observation, ``Has dir proc site`` -> Spec Anatomic Site, ``Has dir device`` ->
Device. ``KnowledgeGraph.edges``/``iter_edges`` (and so ``traverse``) used to
default ``within_domain=True``, which silently dropped all of them. The seed
mirrors the real edges of SNOMED 40482788 "Total replacement of right knee
joint".
"""

from __future__ import annotations

import csv
from datetime import date
from importlib.resources import files
from typing import Iterator, cast

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from orm_loader.helpers import Base
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship

from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.traverse import traverse

KNEE_REPLACEMENT = 40482788
PARENT_PROCEDURE = 4327580
REPAIR_ACTION = 4117981
KNEE_PROSTHESIS = 4030970
RIGHT_KNEE_JOINT = 42539525

CONCEPTS = {
    KNEE_REPLACEMENT: "Procedure",
    PARENT_PROCEDURE: "Procedure",
    REPAIR_ACTION: "Observation",
    KNEE_PROSTHESIS: "Device",
    RIGHT_KNEE_JOINT: "Spec Anatomic Site",
}
ATTRIBUTE_EDGES = {
    ("Has method", REPAIR_ACTION, "Procedural"),
    ("Has dir device", KNEE_PROSTHESIS, "Procedural"),
    ("Has dir proc site", RIGHT_KNEE_JOINT, "Anatomical"),
}


@pytest.fixture(scope="module")
def procedure_kg() -> Iterator[KnowledgeGraph]:
    engine = sa.create_engine("sqlite+pysqlite:///:memory:").execution_options(
        schema_translate_map={"primary": None, "vocab": None}
    )
    tables = cast(
        list[sa.Table],
        [
            Concept.__table__,
            Concept_Relationship.__table__,
            RelationshipClass.__table__,
            RelationshipMapping.__table__,
        ],
    )
    Base.metadata.create_all(engine, tables=tables)
    start, end = date(2020, 1, 1), date(2099, 12, 31)
    session_factory = sessionmaker(bind=engine, future=True)
    with session_factory() as session:
        session.add_all(
            Concept(
                concept_id=concept_id,
                concept_name=f"concept {concept_id}",
                domain_id=domain_id,
                vocabulary_id="SNOMED",
                concept_class_id="Procedure",
                standard_concept="S",
                concept_code=str(concept_id),
                valid_start_date=start,
                valid_end_date=end,
                invalid_reason=None,
            )
            for concept_id, domain_id in CONCEPTS.items()
        )
        edges = {*ATTRIBUTE_EDGES, ("Is a", PARENT_PROCEDURE, "Taxonomic")}
        session.add_all(
            Concept_Relationship(
                concept_id_1=KNEE_REPLACEMENT,
                relationship_id=relationship_id,
                concept_id_2=object_id,
                valid_start_date=start,
                valid_end_date=end,
                invalid_reason=None,
            )
            for relationship_id, object_id, _ in edges
        )
        subkinds = {
            (PredicateKind.ATTRIBUTE, "Procedural"),
            (PredicateKind.ATTRIBUTE, "Anatomical"),
            (PredicateKind.HIERARCHY, "Taxonomic"),
        }
        session.add_all(
            RelationshipClass(
                predicate_kind=kind,
                predicate_subkind=subkind,
                description=subkind,
                semantics=subkind,
                inference="none",
            )
            for kind, subkind in subkinds
        )
        session.add_all(
            RelationshipMapping(
                relationship_id=relationship_id,
                predicate_kind=PredicateKind.HIERARCHY if relationship_id == "Is a" else PredicateKind.ATTRIBUTE,
                predicate_subkind=subkind,
            )
            for relationship_id, _, subkind in edges
        )
        session.commit()
    try:
        yield KnowledgeGraph(session_factory)
    finally:
        engine.dispose()


def _attribute_pairs(edges) -> set[tuple[str, int]]:
    return {(e.predicate_id, e.object_id) for e in edges if e.predicate_kind == PredicateKind.ATTRIBUTE}


EXPECTED_ATTRIBUTE_PAIRS = {(rel, obj) for rel, obj, _ in ATTRIBUTE_EDGES}


def test_edges_include_cross_domain_attributes_by_default(procedure_kg: KnowledgeGraph) -> None:
    edges = procedure_kg.edges(KNEE_REPLACEMENT, direction="out")

    assert _attribute_pairs(edges) == EXPECTED_ATTRIBUTE_PAIRS
    assert ("Is a", PARENT_PROCEDURE) in {(e.predicate_id, e.object_id) for e in edges}


def test_attribute_kind_filter_returns_cross_domain_attributes(procedure_kg: KnowledgeGraph) -> None:
    edges = procedure_kg.edges(
        KNEE_REPLACEMENT,
        direction="out",
        predicate_kinds=frozenset({PredicateKind.ATTRIBUTE}),
    )

    assert {(e.predicate_id, e.object_id) for e in edges} == EXPECTED_ATTRIBUTE_PAIRS


def test_within_domain_opt_in_still_filters(procedure_kg: KnowledgeGraph) -> None:
    edges = procedure_kg.edges(KNEE_REPLACEMENT, direction="out", within_domain=True)

    assert {(e.predicate_id, e.object_id) for e in edges} == {("Is a", PARENT_PROCEDURE)}


def test_traverse_reaches_cross_domain_attributes(procedure_kg: KnowledgeGraph) -> None:
    subgraph, _ = traverse(
        procedure_kg,
        seeds=(KNEE_REPLACEMENT,),
        predicate_kinds={PredicateKind.ATTRIBUTE},
        max_depth=1,
        on=None,
        max_nodes=None,
        trace=False,
    )

    assert {REPAIR_ACTION, KNEE_PROSTHESIS, RIGHT_KNEE_JOINT} <= set(subgraph.nodes)


# SNOMED defining attributes and their inverses, by (kind, subkind) expected in
# the shipped predicate mapping.
DEFINING_ATTRIBUTES = {
    "Has dir proc site": ("Attribute", "Anatomical"),
    "Dir proc site of": ("Attribute", "Anatomical"),
    "Has indir proc site": ("Attribute", "Anatomical"),
    "Indir proc site of": ("Attribute", "Anatomical"),
    "Has proc site": ("Attribute", "Anatomical"),
    "Proc site of": ("Attribute", "Anatomical"),
    "Has finding site": ("Attribute", "Anatomical"),
    "Finding site of": ("Attribute", "Anatomical"),
    "Has specimen topo": ("Attribute", "Anatomical"),
    "Specimen topo of": ("Attribute", "Anatomical"),
    "Has proc context": ("Attribute", "Context"),
    "Proc context of": ("Attribute", "Context"),
    "Has occurrence": ("Attribute", "Temporal"),
    "Occurrence of": ("Attribute", "Temporal"),
    "Using subst": ("Attribute", "Procedural"),
    "Subst used by": ("Attribute", "Procedural"),
    "Has dir subst": ("Attribute", "Procedural"),
    "Dir subst of": ("Attribute", "Procedural"),
    "Has method": ("Attribute", "Procedural"),
    "Has dir device": ("Attribute", "Procedural"),
    "Has access": ("Attribute", "Procedural"),
    "Using acc device": ("Attribute", "Procedural"),
}


def test_shipped_mapping_classes_snomed_defining_attributes_as_attributes() -> None:
    mapping_file = files("omop_graph") / "data" / "predicate_mapping.csv"
    with mapping_file.open(newline="") as handle:
        classes = {row["r_id"]: (row["class"], row["subclass"]) for row in csv.DictReader(handle)}

    assert {r_id: classes.get(r_id) for r_id in DEFINING_ATTRIBUTES} == DEFINING_ATTRIBUTES
