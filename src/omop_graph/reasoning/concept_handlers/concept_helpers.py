"""
Helper functions for handling OMOP Concepts in reasoning pipelines.

This module provides utilities for standardizing concept IDs (mapping non-standard
concepts to their standard counterparts) using the Knowledge Graph.
"""

from typing import Dict, Set

# Local Application Imports
from omop_graph.graph.kg import KnowledgeGraph


def standardise_ids(
    ids: Set[int],
    kg: KnowledgeGraph,
) -> Dict[int, int]:
    """
    Map a set of concept IDs to their Standard Concept IDs.

    This function attempts to find a 'Maps to' relationship for each input ID.
    If a 'Maps to' edge exists, the target ID is used (Standard Concept).
    If no such edge exists, the original ID is returned (fallback to self).

    Parameters
    ----------
    ids : set[int]
        A set of OMOP Concept IDs to standardize.
    kg : KnowledgeGraph
        The graph instance used to look up 'Maps to' relationships.

    Returns
    -------
    dict[int, int]
        A dictionary mapping the original input ID to its standardized ID.
    """
    mapping: Dict[int, int] = {}

    raise NotImplementedError(
        "predicate search has changed. Needs to change here too. Subsumes no longer valid."
    )
    for cid in ids:
        mapped = None
        # Look for the first 'Maps to' relationship
        with kg.session_factory() as session:
            for e in kg.iter_edges(
                session=session,
                concept_ids=cid,
                direction="out",
                predicate="Maps to",
            ):
                mapped = e.object_id
                break  # Assume the first mapping is sufficient

        # Use the mapped ID if found, otherwise keep the original
        mapping[cid] = mapped if mapped is not None else cid

    return mapping
