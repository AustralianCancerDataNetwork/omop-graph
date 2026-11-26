"""Explanations of graph paths: the concept a path resolves to and how each step was found."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omop_graph.extensions.relationship_classification import PredicateKind
from omop_graph.graph.model.matches import LabelMatchKind
from omop_graph.graph.paths import GraphPath, PathStep
from omop_graph.graph.traverse import GraphTrace, TraceStep

if TYPE_CHECKING:
    from omop_graph.graph.kg import KnowledgeGraph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathProfile:
    """
    Represents the resolved 'Anchor Concept' discovered along a graph path.

    Attributes
    ----------
    concept_id : int
        The ID of the resolved concept.
    concept_name : str
        The name of the resolved concept.
    is_standard : bool
        True if `concept_id` is a Standard OMOP Concept.
    original_concept_id : int
        The ID of the starting node (candidate).
    original_concept_name : str
        The name of the starting node.
    path : GraphPath
        The full topological path.
    """

    concept_id: int
    concept_name: str
    is_standard: bool
    original_concept_id: int
    original_concept_name: str
    path: GraphPath

    def __repr__(self) -> str:
        return f"PathProfile(concept_id={self.concept_id} [{self.concept_name}])"

    @classmethod
    def from_path(
        cls,
        kg: KnowledgeGraph,
        path: GraphPath,
        match_kind: LabelMatchKind,
        source_concept_id: int | None = None,
    ) -> PathProfile:
        """
        Analyze a path to determine the 'Standard Anchor'.

        The first Standard Concept encountered via an IDENTITY edge is promoted as
        the anchor.

        Notes
        -----
        For zero-hop paths (source == target), ``source_concept_id``
        must be provided; a ``ValueError`` is raised otherwise.

        Parameters
        ----------
        source_concept_id : int, optional
            Required when ``path`` has no steps (i.e. source == target).
        """
        if not path.steps:
            if source_concept_id is None:
                raise ValueError(
                    "source_concept_id is required for zero-hop paths "
                    "(find_shortest_paths was called with source == target)."
                )
            view = kg.concept_view(source_concept_id)
            return cls(
                concept_id=source_concept_id,
                concept_name=view.concept_name,
                is_standard=view.standard_concept,
                original_concept_id=source_concept_id,
                original_concept_name=view.concept_name,
                path=path,
            )

        node_ids = path.nodes()
        view_map = {v.concept_id: v for v in kg.concept_views(node_ids)}

        def get_view(idx):
            return view_map[node_ids[idx]]

        predicate_kinds = kg.predicate_kinds(tuple(p.predicate for p in path.steps))
        standard_anchor: tuple[int, str] | None = None

        for step_idx in range(len(path.steps)):
            next_view = get_view(step_idx + 1)
            if (
                predicate_kinds[step_idx] is PredicateKind.IDENTITY
                and not standard_anchor
                and next_view.standard_concept
            ):
                standard_anchor = (next_view.concept_id, next_view.concept_name)

        first_view = get_view(0)

        if standard_anchor is None:
            concept_id = first_view.concept_id
            concept_name = first_view.concept_name
            is_standard = first_view.standard_concept
        else:
            concept_id, concept_name = standard_anchor
            is_standard = True

        return cls(
            concept_id=concept_id,
            concept_name=concept_name,
            is_standard=is_standard,
            original_concept_id=first_view.concept_id,
            original_concept_name=first_view.concept_name,
            path=path,
        )


@dataclass(frozen=True)
class PathExplanationStep:
    """
    A single step in the explanation of a path.
    """

    step: PathStep
    traversal_depth: int | None
    predicate_kind: PredicateKind
    reason: str


@dataclass(frozen=True)
class PathExplanation:
    """
    A full explanation of a graph path, including semantic reasoning.
    """

    path: GraphPath
    profile: PathProfile
    steps: tuple[PathExplanationStep, ...]

    @classmethod
    def from_path(
        cls,
        kg: KnowledgeGraph,
        path: GraphPath,
        trace: GraphTrace,
        match_kind: LabelMatchKind,
    ) -> PathExplanation:
        """
        Construct an explanation by combining the path, the trace log, and semantic profiles.
        """
        steps: list[PathExplanationStep] = []
        source = (
            path.steps[0].subject.concept_id
            if path.steps
            else (trace.seeds[0] if trace.seeds else None)
        )
        profile = PathProfile.from_path(kg, path, match_kind=match_kind, source_concept_id=source)

        for step in path.steps:
            ts = trace_contains_step(trace, step)
            kind = kg.predicate_kind(step.predicate)
            reason = kind.value
            steps.append(
                PathExplanationStep(
                    step=step,
                    traversal_depth=ts.depth if ts else None,
                    predicate_kind=kind,
                    reason=reason,
                )
            )
        return cls(
            path=path,
            profile=profile,
            steps=tuple(steps),
        )


def trace_contains_step(trace: GraphTrace, step: PathStep) -> TraceStep | None:
    """
    Check if a specific path step appears in the search trace.
    """
    for ts in trace.steps:
        if ts.node != step.subject.concept_id:
            continue
        for e in ts.expanded_edges:
            if e.object_id == step.object.concept_id and e.predicate_id == step.predicate:
                return ts
    return None
