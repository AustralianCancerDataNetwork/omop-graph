from __future__ import annotations

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPTextAnnotatorInterface


def test_annotate_text_standardizes_non_standard_candidate_without_parent_annotation(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    interface = OMOPTextAnnotatorInterface(kg=mock_cdm_kg)

    annotations = list(
        interface.annotate_text(text="Kidney carcinoma term", annotations=None)
    )

    assert len(annotations) == 1
    assert annotations[0].object_id == "OMOP:196653"
