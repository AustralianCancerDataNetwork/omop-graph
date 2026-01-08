import sqlalchemy as sa
import sqlalchemy.orm as so
from collections import defaultdict
from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Relationship,
)
from omop_graph.graph.kg import KnowledgeGraph


def standardise_ids(
    ids: set[int],
    kg: KnowledgeGraph,
) -> dict[int, int]:
    """
    Map id -> standard_id using KG ('Maps to'), fallback to self.
    """
    mapping: dict[int, int] = {}

    for cid in ids:
        mapped = None
        for e in kg.iter_edges(
            cid,
            direction="out",
            predicate="Maps to",
        ):
            mapped = e.object_id
            break  

        mapping[cid] = mapped if mapped is not None else cid

    return mapping

