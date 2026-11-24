"""
Helper functions for handling OMOP Concepts in reasoning pipelines.

This module provides utilities for standardizing concept IDs (mapping non-standard
concepts to their standard counterparts) using the Knowledge Graph.
"""

# Local Application Imports
from omop_graph.graph.kg import KnowledgeGraph


def standardise_ids(
    ids: set[int],
    kg: KnowledgeGraph,
) -> dict[int, int]:
    """
    Map a set of concept IDs to a Standard Concept ID.

    A concept that stands for several standard concepts maps to the lowest id among them,
    and one that stands for none maps to itself.

    Parameters
    ----------
    ids : set[int]
        A set of OMOP Concept IDs to standardize.
    kg : KnowledgeGraph
        The graph instance used to look up identity relationships.

    Returns
    -------
    dict[int, int]
        A dictionary mapping the original input ID to its standardized ID.
    """
    return {
        concept_id: targets[0] if targets else concept_id
        for concept_id, targets in kg.standardise(ids).items()
    }
