"""Every kind of label match renders in a notebook."""

from __future__ import annotations

import pytest

from omop_graph.graph.nodes import LabelMatch, LabelMatchGroupView, LabelMatchKind


def match(kind: LabelMatchKind) -> LabelMatch:
    return LabelMatch(
        input_query="query",
        matched_concept_label="Label",
        matched_concept_id=1,
        match_kind=kind,
        is_standard=True,
        is_active=True,
        synonym=False,
    )


@pytest.mark.parametrize("kind", LabelMatchKind)
def test_a_match_renders_its_kind_badge(kind: LabelMatchKind) -> None:
    html = match(kind)._repr_html_()

    assert f">{kind.badge}</span>" in html
    assert f"background:{kind.colour}" in html


@pytest.mark.parametrize("kind", LabelMatchKind)
def test_a_group_summary_gives_the_reason_of_its_kind(kind: LabelMatchKind) -> None:
    html = LabelMatchGroupView.from_matches([match(kind)])._repr_html_()

    assert kind.reason in html
    assert "unknown" not in html


def test_the_kinds_are_ranked_by_value() -> None:
    assert sorted(LabelMatchKind, reverse=True) == sorted(
        LabelMatchKind, key=lambda kind: -kind.value
    )
    assert LabelMatchKind.EXACT < LabelMatchKind.STARTS_WITH < LabelMatchKind.EMBEDDING
