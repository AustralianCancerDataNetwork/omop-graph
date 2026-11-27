"""Methods that take many ids give the same answer when they must split them into batches."""

from __future__ import annotations

import inspect
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
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary
from sqlalchemy import event

from omop_graph.extensions.relationship_classification import (
    HierarchyDirection,
    PredicateKind,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.model.edges import EdgeFilter
from omop_graph.graph.model.language import OMOPLanguage
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
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - down"),
        relationship_class(PredicateKind.IDENTITY, "Version lineage"),
        classified(
            "Is a",
            PredicateKind.HIERARCHY,
            "Taxonomic - up",
            generalises=True,
            direction=HierarchyDirection.UP,
        ),
        classified(
            "Subsumes",
            PredicateKind.HIERARCHY,
            "Taxonomic - down",
            direction=HierarchyDirection.DOWN,
        ),
        classified("Concept replaced by", PredicateKind.IDENTITY, "Version lineage"),
        Vocabulary(vocabulary_id="SNOMED", vocabulary_name="SNOMED", vocabulary_concept_id=0),
        *(edge(child, "Is a", parent) for child, parent in parents.items()),
        *(edge(parent, "Subsumes", child) for child, parent in parents.items()),
        edge(DELETED, "Concept replaced by", ROOT),
        *(ancestor(concept_id, concept_id, 0) for concept_id in IDS[:-1]),
        *(ancestor(parent, child) for child, parent in parents.items()),
        *(synonym(concept_id, f"Alias {concept_id}", OMOPLanguage.SPANISH) for concept_id in IDS),
        *(synonym(concept_id, f"Other {concept_id}", OMOPLanguage.ENGLISH) for concept_id in IDS),
    ]
    with sqlite_cdm(rows) as sessions:
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


def _edge_triples(edges) -> set[tuple[int, str, int]]:
    return {(edge.subject_id, edge.predicate_id, edge.object_id) for edge in edges}


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
    "closure_up": lambda a: a.kg.closure(IDS[:-1]),
    "closure_down": lambda a: a.kg.closure(IDS[:-1], HierarchyDirection.DOWN),
    "closure_standardised": lambda a: a.kg.closure(IDS[:-1], standardise=True),
    "standardise": lambda a: a.kg.standardise(IDS),
    "standard_concept": lambda a: a.kg.standard_concept(IDS),
    "standard_mappings": lambda a: a.kg.standard_mappings(IDS),
    "replacements": lambda a: a.kg.replacements(IDS),
    "mappings": lambda a: list(a.kg.mappings(IDS)),
    "translations": lambda a: a.kg.translations(IDS, OMOPLanguage.SPANISH),
    "missing_concepts": lambda a: a.kg.missing_concepts((*IDS, 999999, 999998)),
    "information_content": lambda a: a.kg.information_content(IDS[:-1]),
    "num_ancestors": lambda a: a.kg.get_num_ancestors(IDS),
    "potential_ancestors": lambda a: {
        child: set(parents)
        for child, parents in a.kg.get_potential_ancestors_batch((LEFT_LEAF,), IDS).items()
    },
    "entailed_by_subject": lambda a: set(a.kg.entailed_relationships(IDS, None)),
    "entailed_by_object": lambda a: set(a.kg.entailed_relationships(None, IDS)),
    "entailed_between": lambda a: set(a.kg.entailed_relationships(IDS, IDS)),
    "gap_filled": lambda a: a.kg.gap_filled_relationships(IDS[:-1]),
    "edges_between": lambda a: _edge_triples(a.kg.edges_between(IDS, IDS)),
    "search_identifiers": lambda a: list(a.kg.search("no such name", identifiers=IDS)),
}

ID_CALLS = {
    "concept_views",
    "synonyms",
    "relationships_by_subject",
    "relationships_by_object",
    "edges_out",
    "edges_in",
    "closure_up",
    "closure_down",
    "standardise",
    "standard_concept",
    "standard_mappings",
    "replacements",
    "mappings",
    "translations",
    "missing_concepts",
    "information_content",
    "num_ancestors",
    "potential_ancestors",
    "entailed_by_subject",
    "entailed_by_object",
    "gap_filled",
    "edges_between",
}
"""The calls that take ids, and whose statements bind fewer of them in a batch."""

SWEPT_METHODS = {
    "closure",
    "concept_views",
    "edges",
    "edges_between",
    "entailed_relationships",
    "gap_filled_relationships",
    "get_num_ancestors",
    "get_potential_ancestors_batch",
    "information_content",
    "mappings",
    "missing_concepts",
    "relationships",
    "replacements",
    "search",
    "standard_concept",
    "standard_mappings",
    "standardise",
    "synonyms",
    "translations",
}
UNSWEPT_METHODS = {
    "concept_embeddings",
    "concepts_without_embedding",
    "embed_concepts",
    "iter_edges",
    "require_concepts",
}
"""The methods that need an embedding store, batched and tested in test_embedding_optional, and the one that checks only."""


def test_every_method_that_takes_many_ids_is_swept() -> None:
    parameters = {
        "concept_ids",
        "child_ids",
        "parent_ids",
        "subjects",
        "objects",
        "subject_ids",
        "object_ids",
        "identifiers",
    }
    taking_ids = {
        name
        for name, method in inspect.getmembers(KnowledgeGraph, inspect.isfunction)
        if not name.startswith("_") and parameters & set(inspect.signature(method).parameters)
    }

    assert taking_ids - UNSWEPT_METHODS == SWEPT_METHODS


@pytest.mark.parametrize("name", CALLS)
def test_a_method_gives_the_same_answer_when_it_splits_the_ids_into_batches(
    adapter: OMOPAlchemyImplementation, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    call = CALLS[name]
    whole = as_comparable(call(adapter))

    monkeypatch.setattr(KnowledgeGraph, "LOOKUP_BATCH_SIZE", TINY_BATCH)

    assert as_comparable(call(adapter)) == whole


def _largest_binding(adapter: OMOPAlchemyImplementation, call: Callable[..., Any]) -> int:
    """The most parameters any statement of *call* binds."""
    bound: list[int] = []
    with adapter.kg.session_factory() as session:
        engine = session.get_bind()

    def record(conn, cursor, statement, parameters, *args):
        bound.append(len(parameters))

    event.listen(engine, "before_cursor_execute", record)
    try:
        call(adapter)
    finally:
        event.remove(engine, "before_cursor_execute", record)
    return max(bound, default=0)


@pytest.mark.parametrize("name", [name for name in CALLS if name in ID_CALLS])
def test_a_batch_binds_fewer_ids_than_the_whole_list(
    adapter: OMOPAlchemyImplementation, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    whole = _largest_binding(adapter, CALLS[name])
    monkeypatch.setattr(KnowledgeGraph, "LOOKUP_BATCH_SIZE", TINY_BATCH)

    assert _largest_binding(adapter, CALLS[name]) < whole


def test_the_identifiers_of_a_search_are_looked_up_in_batches(
    adapter: OMOPAlchemyImplementation, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(KnowledgeGraph, "LOOKUP_BATCH_SIZE", TINY_BATCH)
    in_lists: list[int] = []
    with adapter.kg.session_factory() as session:
        engine = session.get_bind()

    def record(conn, cursor, statement, parameters, *args):
        if "concept_id IN" in statement:
            in_lists.append(len(parameters))

    event.listen(engine, "before_cursor_execute", record)
    try:
        found = list(adapter.kg.search("no such name", identifiers=IDS[::-1]))
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert in_lists
    assert max(in_lists) <= TINY_BATCH
    assert found == list(IDS[::-1])


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
