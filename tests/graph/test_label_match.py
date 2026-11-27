"""Every kind of label match renders in a notebook."""

from __future__ import annotations

import pytest

from omop_graph.graph.model.matches import LabelMatch, LabelMatchGroupView, LabelMatchKind


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


def test_the_kinds_rank_from_the_strictest_match_to_the_loosest() -> None:
    assert sorted(LabelMatchKind) == [
        LabelMatchKind.EXACT,
        LabelMatchKind.STARTS_WITH,
        LabelMatchKind.PARTIAL,
        LabelMatchKind.FTS,
        LabelMatchKind.EMBEDDING,
    ]


def test_no_two_kinds_share_a_priority() -> None:
    assert [kind.name for kind in LabelMatchKind] == list(LabelMatchKind.__members__)


def test_the_default_resolvers_run_in_the_order_of_the_kinds() -> None:
    from omop_graph.reasoning.resolvers.resolvers import ALL_RESOLVERS

    order = [(resolver.match_kind, resolver.synonym) for resolver in ALL_RESOLVERS]

    assert order == sorted(order)
    assert [kind for kind, _ in order[:4]] == [LabelMatchKind.EXACT] * 2 + [
        LabelMatchKind.PARTIAL
    ] * 2
