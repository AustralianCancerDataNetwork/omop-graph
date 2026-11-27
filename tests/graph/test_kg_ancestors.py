from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.chain_cdm import HOLDER, HOLDER_LEAF, HOLDER_MIDDLE, chain_cdm

from omop_graph.graph.capabilities.concepts import UnknownConceptError
from omop_graph.graph.kg import KnowledgeGraph


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with chain_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_the_ancestors_are_those_of_every_level_without_the_concept_itself(kg) -> None:
    assert kg.ancestors(HOLDER_LEAF) == (HOLDER, HOLDER_MIDDLE)
    assert kg.ancestors(HOLDER) == ()
    assert kg.parents(HOLDER_LEAF) == (HOLDER_MIDDLE,)


def test_the_descendants_are_those_of_every_level_without_the_concept_itself(kg) -> None:
    assert kg.descendants(HOLDER) == (HOLDER_MIDDLE, HOLDER_LEAF)
    assert kg.descendants(HOLDER_LEAF) == ()
    assert kg.children(HOLDER) == (HOLDER_MIDDLE,)


def test_the_strict_hierarchy_is_asked_for_with_isa_only(kg) -> None:
    assert kg.ancestors(HOLDER_LEAF, isa_only=True) == (HOLDER, HOLDER_MIDDLE)
    assert kg.descendants(HOLDER, isa_only=True) == (HOLDER_MIDDLE, HOLDER_LEAF)


def test_an_unknown_concept_has_no_strict_ancestors(kg) -> None:
    with pytest.raises(UnknownConceptError):
        kg.ancestors(999999, isa_only=True)

    assert kg.ancestors(999999) == ()
