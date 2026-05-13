from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, cast

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatch, LabelMatchKind
from omop_graph.reasoning.resolvers.resolver_pipeline import ResolverPipeline
from omop_graph.reasoning.resolvers.resolvers import (
    ExactLabelResolver,
    ExactSynonymResolver,
    FullTextResolver,
    PartialLabelResolver,
)


@dataclass
class _Case:
    text: str
    hits: Dict[str, List[int]]


class _KG:
    def __init__(self, case: _Case) -> None:
        self.case = case

    def concept_lookup(self, label: str, match_kind: LabelMatchKind, synonym: bool = False, search_constraint=None, sort: bool = False):
        key = _key(match_kind, synonym)
        concept_ids = self.case.hits.get(key, [])
        return tuple(
            LabelMatch(
                input_label=label,
                matched_label=f"concept-{cid}",
                concept_id=cid,
                match_kind=match_kind,
                is_standard=True,
                is_active=True,
                synonym=synonym,
            )
            for cid in concept_ids
        )


def _key(match_kind: LabelMatchKind, synonym: bool) -> str:
    if match_kind == LabelMatchKind.EXACT:
        return "exact_synonym" if synonym else "exact"
    if match_kind == LabelMatchKind.PARTIAL:
        return "partial_synonym" if synonym else "partial"
    if match_kind == LabelMatchKind.FTS:
        return "fts_synonym" if synonym else "fts"
    raise ValueError("Unexpected match kind")


def test_exact_and_synonym_resolvers_pull_expected_hits():
    case = _Case(
        text="Neoplasm of kidney",
        hits={
            "exact": [],
            "exact_synonym": [196653],
            "partial": [999001],
            "partial_synonym": [],
            "fts": [],
            "fts_synonym": [],
        },
    )
    kg = _KG(case)
    typed_kg = cast(KnowledgeGraph, kg)

    exact = ExactLabelResolver()
    synonym = ExactSynonymResolver()

    exact_hits = list(exact.resolve(typed_kg, case.text))
    synonym_hits = list(synonym.resolve(typed_kg, case.text))

    assert len(exact_hits) == 0
    assert [h.concept_id for h in synonym_hits] == [196653]


def test_pipeline_deduplicates_hits_across_resolvers():
    case = _Case(
        text="Kidney cancer",
        hits={
            "exact": [196653],
            "exact_synonym": [196653],
            "partial": [196653, 999001],
            "partial_synonym": [196653],
            "fts": [196653],
            "fts_synonym": [],
        },
    )
    kg = _KG(case)
    typed_kg = cast(KnowledgeGraph, kg)

    pipeline = ResolverPipeline(
        resolvers=(
            ExactLabelResolver(),
            ExactSynonymResolver(),
            PartialLabelResolver(),
            FullTextResolver(),
        )
    )

    predictions = [h.concept_id for h in pipeline.resolve(typed_kg, case.text)]

    assert predictions.count(196653) == 1
    assert 999001 in predictions


def test_pipeline_stop_after_resolver_honored():
    case = _Case(
        text="ambiguous term",
        hits={
            "exact": [],
            "exact_synonym": [],
            "partial": [333002],
            "partial_synonym": [],
            "fts": [333001],
            "fts_synonym": [],
        },
    )
    kg = _KG(case)
    typed_kg = cast(KnowledgeGraph, kg)

    pipeline = ResolverPipeline(
        resolvers=(ExactLabelResolver(), PartialLabelResolver(), FullTextResolver()),
        stop_after_resolver=PartialLabelResolver,
    )

    predictions = [h.concept_id for h in pipeline.resolve(typed_kg, case.text)]

    assert 333002 in predictions
    assert 333001 not in predictions
