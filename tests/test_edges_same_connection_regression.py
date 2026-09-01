"""Same-connection regression coverage for kg.py's split-vocab merge (Phase 3.2).

``KnowledgeGraph.iter_edges``/``predicate``/``predicates`` gained a
split-connection branch (see ``test_vocab_split_connection.py``). This pins
the default, unsplit path -- the one every existing deployment actually
uses -- stays on the original single eager join, protecting against a
future edit accidentally forcing the split-path branch unconditionally.
"""

from __future__ import annotations

from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.kg import KnowledgeGraph


def test_edges_use_single_eager_join_when_no_split_is_configured(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    assert mock_cdm_kg._vocab_split is False

    edges = mock_cdm_kg.edges(
        concept_ids=900001,
        direction="out",
        active_only=False,
        within_domain=False,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.subject_id == 900001
    assert edge.object_id == 196653
    assert edge.predicate_id == "maps to"
    assert edge.predicate_kind == PredicateKind.IDENTITY
    assert edge.predicate_subkind == "mapping"
