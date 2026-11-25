"""Methods that take many ids give the same answer when they must split them into batches."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fixtures.cdm_builders import (
    ancestor,
    classified,
    concept,
    edge,
    relationship,
    relationship_class,
    sqlite_cdm,
    synonym,
)
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.concept_synonym import Concept_Synonym
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary
from sqlalchemy import event

from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.edges import EdgeFilter
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.codec import OMOPLanguage
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

ROOT, LEFT, RIGHT, LEFT_LEAF, RIGHT_LEAF, LONELY, DELETED = range(500, 507)
IDS = (ROOT, LEFT, RIGHT, LEFT_LEAF, RIGHT_LEAF, LONELY, DELETED)
TINY_BATCH = 2


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    parents = {LEFT: ROOT, RIGHT: ROOT, LEFT_LEAF: LEFT, RIGHT_LEAF: RIGHT}
    rows = [
        *(concept(concept_id) for concept_id in IDS[:-1]),
        concept(DELETED, invalid_reason="D"),
        relationship("Is a", "Subsumes"),
        relationship("Subsumes", "Is a"),
        relationship("Concept replaced by", "Concept replaces"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up", HierarchyDirection.UP),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - down", HierarchyDirection.DOWN),
        relationship_class(PredicateKind.IDENTITY, "Version lineage"),
        classified("Is a", PredicateKind.HIERARCHY, "Taxonomic - up", generalises=True),
        classified("Subsumes", PredicateKind.HIERARCHY, "Taxonomic - down"),
        classified("Concept replaced by", PredicateKind.IDENTITY, "Version lineage"),
        *(edge(child, "Is a", parent) for child, parent in parents.items()),
        *(edge(parent, "Subsumes", child) for child, parent in parents.items()),
        edge(DELETED, "Concept replaced by", ROOT),
        *(ancestor(concept_id, concept_id, 0) for concept_id in IDS[:-1]),
        *(ancestor(parent, child) for child, parent in parents.items()),
        *(synonym(concept_id, f"Alias {concept_id}", OMOPLanguage.SPANISH) for concept_id in IDS),
        *(synonym(concept_id, f"Other {concept_id}", OMOPLanguage.ENGLISH) for concept_id in IDS),
    ]
    models = (
        Concept,
        Vocabulary,
        Relationship,
        Concept_Ancestor,
        Concept_Relationship,
        Concept_Synonym,
        RelationshipClass,
        RelationshipMapping,
    )
    with sqlite_cdm(models, rows) as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


def curies(ids: tuple[int, ...] = IDS) -> list[str]:
    return [f"OMOP:{concept_id}" for concept_id in ids]


def as_comparable(result: Any) -> Any:
    """A result made comparable: unordered collections as sets, oaklib objects as reprs."""
    if isinstance(result, dict):
        return {key: as_comparable(value) for key, value in result.items()}
    if isinstance(result, (set, frozenset)):
        return {as_comparable(value) for value in result}
    if isinstance(result, tuple):
        return tuple(as_comparable(value) for value in result)
    if isinstance(result, list):
        return [as_comparable(value) for value in result]
    return result if isinstance(result, (str, int, float, bool, type(None))) else repr(result)


CALLS: dict[str, Callable[[OMOPAlchemyImplementation], Any]] = {
    "concept_views": lambda a: sorted(v.concept_id for v in a.kg.concept_views(IDS)),
    "concept_views_unsorted": lambda a: {v.concept_id for v in a.kg.concept_views(IDS, sort=False)},
    "synonyms": lambda a: a.kg.synonyms(IDS),
    "relationships_by_subject": lambda a: set(a.kg.relationships(IDS)),
    "relationships_by_object": lambda a: set(a.kg.relationships(None, objects=IDS)),
    "relationships_by_both": lambda a: set(a.kg.relationships(IDS, objects=IDS)),
    "relationships_inverted": lambda a: set(a.kg.relationships(IDS, invert=True)),
    "edges_out": lambda a: {
        (e.subject_id, e.predicate_id, e.object_id) for e in a.kg.edges(IDS, "out")
    },
    "edges_in": lambda a: {
        (e.subject_id, e.predicate_id, e.object_id) for e in a.kg.edges(IDS, "in")
    },
    "entities": lambda a: set(a.kg.entities()),
    "roots": lambda a: set(a.kg.roots()),
    "leaves": lambda a: set(a.kg.leaves()),
    "singletons": lambda a: set(a.kg.singletons()),
    "labels": lambda a: list(a.labels(curies())),
    "translated_labels": lambda a: list(a.labels(curies(), lang="es")),
    "multilingual_labels": lambda a: set(a.multilingual_labels(curies())),
    "defined_bys": lambda a: list(a.defined_bys(curies())),
    "adapter_relationships": lambda a: set(a.relationships(subjects=curies())),
    "entailed_relationships": lambda a: set(
        a.relationships(subjects=curies(), include_entailed=True)
    ),
    "all_relationships": lambda a: set(a.relationships()),
    "relationships_metadata": lambda a: [
        metadata for _, metadata in a.relationships_metadata(a.relationships(subjects=curies()))
    ],
    "entity_metadata": lambda a: {c: a.entity_metadata_map(c) for c in curies()},
    "nodes": lambda a: [(node.id, node.lbl, node.meta) for node in a.nodes()],
    "node": lambda a: [a.node(c, include_metadata=True) for c in curies()],
}


@pytest.mark.parametrize("name", CALLS)
def test_a_method_gives_the_same_answer_when_it_splits_the_ids_into_batches(
    adapter: OMOPAlchemyImplementation, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    call = CALLS[name]
    whole = as_comparable(call(adapter))

    monkeypatch.setattr(KnowledgeGraph, "LOOKUP_BATCH_SIZE", TINY_BATCH)

    assert as_comparable(call(adapter)) == whole


def test_the_batched_methods_find_something_to_compare(adapter: OMOPAlchemyImplementation) -> None:
    assert len(adapter.kg.concept_views(IDS)) == len(IDS)
    assert len(adapter.kg.synonyms(IDS)) == 2 * len(IDS)
    assert len(tuple(adapter.kg.relationships(IDS))) > TINY_BATCH
    assert len(list(adapter.nodes())) > TINY_BATCH


def test_synonyms_are_ordered_by_concept_across_batches(
    adapter: OMOPAlchemyImplementation, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(KnowledgeGraph, "LOOKUP_BATCH_SIZE", TINY_BATCH)
    concept_ids = [synonym_.concept_id for synonym_ in adapter.kg.synonyms(IDS[::-1])]

    assert concept_ids == sorted(concept_ids)


def test_a_relationship_between_subjects_and_objects_in_different_batches_is_found(
    adapter: OMOPAlchemyImplementation, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(KnowledgeGraph, "LOOKUP_BATCH_SIZE", 1)
    edge_filter = EdgeFilter(predicate_ids=frozenset({"Is a"}))

    found = set(
        adapter.kg.relationships(
            (LEFT_LEAF, RIGHT_LEAF), objects=(LEFT, RIGHT), edge_filter=edge_filter
        )
    )

    assert found == {(LEFT_LEAF, "Is a", LEFT), (RIGHT_LEAF, "Is a", RIGHT)}


def test_no_subjects_or_no_objects_select_no_relationship(
    adapter: OMOPAlchemyImplementation,
) -> None:
    assert list(adapter.kg.relationships(())) == []
    assert list(adapter.kg.relationships(IDS, objects=())) == []


def test_nodes_are_the_nodes_of_each_entity(adapter: OMOPAlchemyImplementation) -> None:
    one_by_one = [adapter.node(c, include_metadata=True) for c in adapter.entities()]

    assert as_comparable(list(adapter.nodes())) == as_comparable(one_by_one)


def test_the_nodes_of_all_concepts_take_a_fixed_number_of_queries(
    adapter: OMOPAlchemyImplementation,
) -> None:
    statements: list[str] = []
    with adapter.kg.session_factory() as session:
        engine = session.get_bind()
    listener = lambda conn, cursor, statement, *args: statements.append(statement)
    event.listen(engine, "before_cursor_execute", listener)
    try:
        nodes = list(adapter.nodes())
    finally:
        event.remove(engine, "before_cursor_execute", listener)

    assert len(nodes) == len(IDS) - 1
    assert len(statements) < 2 * len(nodes)
