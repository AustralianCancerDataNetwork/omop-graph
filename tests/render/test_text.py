from __future__ import annotations

from collections.abc import Iterator

import pytest
from fixtures.hierarchy_cdm import CANCER, LUNG_CANCER, hierarchy_cdm

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.render.text import edge_text


@pytest.fixture(scope="module")
def kg() -> Iterator[KnowledgeGraph]:
    with hierarchy_cdm() as sessions:
        yield KnowledgeGraph(sessions)


def test_an_edge_reads_with_the_names_of_its_concepts_and_relationship(kg) -> None:
    [edge] = [e for e in kg.edges(LUNG_CANCER, "out") if e.object_id == CANCER]

    assert edge_text(kg, edge) == (
        f"{kg.concept_view(LUNG_CANCER).concept_name} -[Is a]-> "
        f"{kg.concept_view(CANCER).concept_name}"
    )
