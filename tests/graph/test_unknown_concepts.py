from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import CANCER, DISEASE, hierarchy_cdm
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.capabilities.concepts import UnknownConceptError
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.structural_similarity import ICSource

MISSING = 999999


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda kg: kg.closure([CANCER, MISSING]), id="closure"),
        pytest.param(lambda kg: kg.standardise([CANCER, MISSING]), id="standardise"),
        pytest.param(lambda kg: kg.standard_mappings([CANCER, MISSING]), id="standard_mappings"),
        pytest.param(
            lambda kg: kg.information_content([CANCER, MISSING]), id="information_content"
        ),
        pytest.param(lambda kg: kg.pairwise_similarity(CANCER, MISSING), id="pairwise_similarity"),
        pytest.param(lambda kg: kg.common_ancestors(CANCER, MISSING), id="common_ancestors"),
        pytest.param(lambda kg: kg.common_descendants(CANCER, MISSING), id="common_descendants"),
        pytest.param(
            lambda kg: kg.most_recent_common_ancestors(CANCER, MISSING),
            id="most_recent_common_ancestors",
        ),
        pytest.param(lambda kg: kg.jaccard_similarity(CANCER, MISSING), id="jaccard_similarity"),
        pytest.param(lambda kg: kg.concept_view(MISSING), id="concept_view"),
    ],
)
def test_working_out_a_property_of_a_missing_concept_names_it(kg, operation) -> None:
    with pytest.raises(UnknownConceptError) as raised:
        operation(kg)

    assert raised.value.concept_ids == (MISSING,)


def test_the_error_reads_as_a_message_without_quotes(kg) -> None:
    with pytest.raises(UnknownConceptError) as raised:
        kg.require_concepts([MISSING, 999998])

    assert str(raised.value) == "These concepts do not exist: [999999, 999998]"


def test_missing_concepts_lists_each_once_in_the_order_given(kg) -> None:
    assert kg.missing_concepts([MISSING, CANCER, 999998, MISSING]) == (MISSING, 999998)
    assert kg.missing_concepts([]) == ()


def test_queries_that_select_rows_by_id_give_no_rows_for_a_missing_concept(kg) -> None:
    assert list(kg.relationships([MISSING])) == []
    assert kg.edges(MISSING, "out") == ()


def test_an_information_content_over_no_concepts_is_refused(kg) -> None:
    nothing = ConceptFilter(domains=("No such domain",))

    with pytest.raises(ValueError, match="selects no concept"):
        kg.information_content([CANCER], concept_filter=nothing)


def test_descendants_outnumbering_the_scope_are_refused_and_not_clipped(kg) -> None:
    narrow = ConceptFilter(concept_ids=(DISEASE,), require_active=True)

    with pytest.raises(ValueError, match="outnumber"):
        kg.information_content([DISEASE], ICSource.CONCEPT_ANCESTOR, narrow)
