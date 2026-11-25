"""Standard concept search from candidates, on a small CDM with each kind of route to a standard concept."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import (
    ancestor,
    classified,
    concept,
    edge_pair,
    relationship,
    relationship_class,
    sqlite_cdm,
)
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from sqlalchemy import event

from omop_graph.extensions.omop_alchemy import (
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.edges import EdgeFilter
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.paths import StandardConcept, find_standard_paths
from omop_graph.reasoning.resolvers import CandidateHit

TARGET_A, TARGET_B, LEAF, DEEP_LEAF = 10, 11, 20, 21
SOURCE, SOURCE_TWO, SOURCE_CHAINED, SOURCE_DANGLING, SOURCE_CROSS, ORPHAN = 30, 31, 32, 33, 34, 35
REPLACED, CHAIN_HEAD, CHAIN_MIDDLE, DRUG = 40, 50, 51, 70
DANGLING = 999
IDENTITY = EdgeFilter(predicate_kinds=frozenset({PredicateKind.IDENTITY}), within_domain=True)


def hit(concept_id: int) -> CandidateHit:
    return CandidateHit(concept_id, LabelMatchKind.EXACT, f"label {concept_id}", False)


def reached(found: list[StandardConcept]) -> set[tuple[int, int, int]]:
    """The concept, separation and identity hops of each result."""
    return {(c.concept_id, c.separation, c.identity_hops) for c in found}


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    drug = concept(DRUG)
    drug.domain_id = "Drug"
    pairs = (
        ("Maps to", "Mapped from", "Exact equivalence"),
        ("Concept replaced by", "Concept replaces", "Version lineage"),
    )
    rows = [
        *(concept(i) for i in (TARGET_A, TARGET_B, LEAF, DEEP_LEAF)),
        *(concept(i, standard_concept=None) for i in (SOURCE, SOURCE_TWO, SOURCE_CHAINED)),
        *(concept(i, standard_concept=None) for i in (SOURCE_DANGLING, SOURCE_CROSS, ORPHAN)),
        *(concept(i, invalid_reason="D") for i in (REPLACED, CHAIN_HEAD, CHAIN_MIDDLE)),
        drug,
        *(
            relationship(r, reverse)
            for forward, backward, _ in pairs
            for r, reverse in ((forward, backward), (backward, forward))
        ),
        relationship_class(PredicateKind.IDENTITY, "Exact equivalence"),
        relationship_class(PredicateKind.IDENTITY, "Version lineage"),
        *(
            classified(r, PredicateKind.IDENTITY, subkind)
            for forward, backward, subkind in pairs
            for r in (forward, backward)
        ),
        ancestor(TARGET_A, LEAF, 2),
        ancestor(TARGET_B, LEAF, 1),
        ancestor(TARGET_A, DEEP_LEAF, 5),
        *edge_pair(SOURCE, "Maps to", "Mapped from", LEAF),
        *edge_pair(SOURCE_TWO, "Maps to", "Mapped from", LEAF),
        *edge_pair(SOURCE_TWO, "Maps to", "Mapped from", DEEP_LEAF),
        *edge_pair(SOURCE_CHAINED, "Maps to", "Mapped from", SOURCE),
        *edge_pair(SOURCE_DANGLING, "Maps to", "Mapped from", DANGLING),
        *edge_pair(SOURCE_CROSS, "Maps to", "Mapped from", DRUG),
        *edge_pair(REPLACED, "Concept replaced by", "Concept replaces", LEAF),
        *edge_pair(CHAIN_HEAD, "Concept replaced by", "Concept replaces", CHAIN_MIDDLE),
        *edge_pair(CHAIN_MIDDLE, "Concept replaced by", "Concept replaces", LEAF),
    ]
    models = (
        Concept,
        Concept_Ancestor,
        Relationship,
        Concept_Relationship,
        RelationshipClass,
        RelationshipMapping,
    )
    with sqlite_cdm(models, rows) as sessions:
        yield KnowledgeGraph(sessions)


def search(kg: KnowledgeGraph, candidates: list[int], targets=None, **kwargs) -> list[set]:
    found = find_standard_paths(
        kg, targets, [hit(c) for c in candidates], edge_filter=IDENTITY, **kwargs
    )
    return [reached(f) for f in found]


def test_a_standard_candidate_is_its_own_standard_concept_at_zero_hops(kg) -> None:
    assert search(kg, [LEAF]) == [{(LEAF, 0, 0)}]


def test_a_non_standard_candidate_walks_one_edge_to_its_standard_concept(kg) -> None:
    assert search(kg, [SOURCE]) == [{(LEAF, 1, 1)}]


def test_a_candidate_with_several_standard_concepts_reaches_them_all(kg) -> None:
    assert search(kg, [SOURCE_TWO]) == [{(LEAF, 1, 1), (DEEP_LEAF, 1, 1)}]


def test_edges_to_non_standard_concepts_are_not_followed(kg) -> None:
    assert search(kg, [SOURCE_CHAINED, ORPHAN]) == [set(), set()]


def test_an_edge_to_a_concept_that_does_not_exist_is_skipped(kg) -> None:
    assert search(kg, [SOURCE_DANGLING]) == [set()]


def test_the_edge_filter_limits_the_edges_walked(kg) -> None:
    assert search(kg, [SOURCE_CROSS]) == [set()]
    found = find_standard_paths(
        kg,
        None,
        [hit(SOURCE_CROSS)],
        edge_filter=EdgeFilter(predicate_kinds=frozenset({PredicateKind.IDENTITY})),
    )
    assert reached(found[0]) == {(DRUG, 1, 1)}


def test_without_targets_max_depth_caps_the_edges_walked(kg) -> None:
    assert search(kg, [SOURCE, LEAF], max_depth=0) == [set(), {(LEAF, 0, 0)}]


def test_with_targets_a_descendant_is_recorded_for_each_target_it_descends_from(kg) -> None:
    assert search(kg, [SOURCE], (TARGET_A, TARGET_B)) == [{(LEAF, 2, 1), (LEAF, 1, 1)}]
    assert search(kg, [SOURCE], (TARGET_B,)) == [{(LEAF, 1, 1)}]


def test_with_targets_a_concept_that_descends_from_none_is_expanded(kg) -> None:
    assert search(kg, [REPLACED], (TARGET_A,)) == [{(LEAF, 2, 1)}]
    assert search(kg, [CHAIN_HEAD], (TARGET_A,)) == [{(LEAF, 2, 2)}]


def test_with_targets_max_depth_caps_the_ancestor_separation(kg) -> None:
    assert search(kg, [DEEP_LEAF], (TARGET_A,), max_depth=4) == [set()]
    assert search(kg, [DEEP_LEAF], (TARGET_A,), max_depth=5) == [{(DEEP_LEAF, 5, 0)}]


def test_max_concepts_caps_each_candidate_apart(kg) -> None:
    found = search(kg, [SOURCE_TWO, SOURCE], max_concepts=1)

    assert [len(f) for f in found] == [1, 1]


def test_candidates_searched_together_give_the_results_of_searching_each_apart(kg) -> None:
    candidates = [SOURCE, REPLACED, ORPHAN, SOURCE_TWO, CHAIN_HEAD, LEAF, SOURCE]

    for targets in (None, (TARGET_A, TARGET_B)):
        together = search(kg, candidates, targets)
        apart = [search(kg, [candidate], targets)[0] for candidate in candidates]
        assert together == apart


def test_results_keep_the_candidate_they_started_from(kg) -> None:
    found = find_standard_paths(kg, None, [hit(SOURCE), hit(ORPHAN)], edge_filter=IDENTITY)

    assert [c.original_id for c in found[0]] == [SOURCE]
    assert found[1] == []
    assert found[0][0].matched_concept_label == f"label {SOURCE}"
    assert found[0][0].concept_name == f"Concept {LEAF}"


def test_the_queries_made_do_not_grow_with_the_candidates(kg) -> None:
    def queries(candidates: list[int], targets) -> int:
        count = 0

        def count_query(*_) -> None:
            nonlocal count
            count += 1

        with kg.session_factory() as session:
            engine = session.get_bind()
        event.listen(engine, "before_cursor_execute", count_query)
        try:
            search(kg, candidates, targets)
        finally:
            event.remove(engine, "before_cursor_execute", count_query)
        return count

    for targets in (None, (TARGET_A,)):
        assert queries([SOURCE] * 1, targets) == queries([SOURCE, SOURCE_TWO] * 20, targets)
