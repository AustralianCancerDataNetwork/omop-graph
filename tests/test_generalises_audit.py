"""The generalises audit script measures inheritance through ``Is a`` and proposes flags."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import cast

import pytest
import sqlalchemy as sa
from omop_alchemy.cdm.model.vocabulary import Concept_Relationship
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

_SCRIPT = Path(__file__).parents[1] / "scripts" / "generalises_audit.py"
_spec = importlib.util.spec_from_file_location("generalises_audit", _SCRIPT)
assert _spec and _spec.loader
audit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = audit
_spec.loader.exec_module(audit)

HAS_SITE, SITE_OF = "Has finding site", "Finding site of"
LUNG_CANCER, SMALL_CELL, ADENO, UNRELATED = 1, 2, 3, 4
LUNG, THORACIC_ORGAN, RIGHT_LUNG, LIVER = 10, 11, 12, 13


def _edge(subject: int, relationship_id: str, obj: int, invalid_reason: str | None = None):
    return Concept_Relationship(
        concept_id_1=subject,
        concept_id_2=obj,
        relationship_id=relationship_id,
        valid_start_date=date(2020, 1, 1),
        valid_end_date=date(2099, 12, 31),
        invalid_reason=invalid_reason,
    )


@pytest.fixture
def session() -> Iterator[Session]:
    """Lung cancer has a site, and its children carry it, specialise it, or do not hold it."""
    engine = sa.create_engine("sqlite+pysqlite:///:memory:").execution_options(
        schema_translate_map={"primary": None, "vocab": None}
    )
    cast(sa.Table, Concept_Relationship.__table__).create(engine)
    with Session(engine) as session:
        session.add_all(
            [
                _edge(SMALL_CELL, "Is a", LUNG_CANCER),
                _edge(ADENO, "Is a", LUNG_CANCER),
                _edge(UNRELATED, "Is a", LUNG_CANCER),
                _edge(RIGHT_LUNG, "Is a", LUNG),
                _edge(LUNG, "Is a", THORACIC_ORGAN),
                _edge(LIVER, "Is a", LIVER, invalid_reason="D"),
                _edge(LUNG_CANCER, HAS_SITE, LUNG),
                _edge(SMALL_CELL, HAS_SITE, LUNG),
                _edge(ADENO, HAS_SITE, RIGHT_LUNG),
                _edge(UNRELATED, HAS_SITE, LIVER),
            ]
        )
        session.commit()
        yield session
    engine.dispose()


def test_the_strict_ancestors_follow_the_valid_is_a_edges_upward(session):
    ancestors = audit.strict_ancestors(session, [RIGHT_LUNG, LIVER])

    assert ancestors[RIGHT_LUNG] == {LUNG, THORACIC_ORGAN}
    assert ancestors[LIVER] == frozenset()


def test_a_cycle_of_is_a_edges_ends_and_never_lists_the_concept_itself(session):
    session.add_all([_edge(THORACIC_ORGAN, "Is a", RIGHT_LUNG)])
    session.commit()

    ancestors = audit.strict_ancestors(session, [RIGHT_LUNG])

    assert ancestors[RIGHT_LUNG] == {LUNG, THORACIC_ORGAN}


def test_a_child_carries_the_parent_object_by_holding_it_or_something_below_it(session):
    measurement = audit.measure(session, HAS_SITE, sample_edges=10, sample_pairs=10)

    assert measurement == audit.Measurement(pairs=3, holders=3, covered=2)


def test_a_child_that_holds_nothing_counts_as_a_pair_and_not_as_a_holder(session):
    session.execute(
        sa.delete(Concept_Relationship).where(
            Concept_Relationship.concept_id_1 == UNRELATED,
            Concept_Relationship.relationship_id == HAS_SITE,
        )
    )
    session.commit()

    measurement = audit.measure(session, HAS_SITE, sample_edges=10, sample_pairs=10)

    assert measurement == audit.Measurement(pairs=3, holders=2, covered=2)


def test_an_invalid_edge_is_not_a_held_object(session):
    session.add(_edge(UNRELATED, HAS_SITE, LUNG, invalid_reason="D"))
    session.commit()

    measurement = audit.measure(session, HAS_SITE, sample_edges=10, sample_pairs=10)

    assert measurement.covered == 2


def _cancelled(sqlstate: str):
    def raise_it(*args, **kwargs):
        error = type("Cancelled", (Exception,), {"sqlstate": sqlstate})("statement")
        raise DBAPIError("statement", {}, error)

    return raise_it


def test_only_a_statement_timeout_is_reported_as_timed_out(session, monkeypatch):
    monkeypatch.setattr(audit, "measure", _cancelled(audit.QUERY_CANCELED))
    assert audit.measure_within(session, HAS_SITE, 10, 10, 1) == audit.TimedOut()

    monkeypatch.setattr(audit, "measure", _cancelled("42P01"))
    with pytest.raises(DBAPIError):
        audit.measure_within(session, HAS_SITE, 10, 10, 1)


SETTINGS = audit.Settings(min_pairs=10, min_share=0.7, max_reverse=0.15)
INHERITS = audit.Measurement(pairs=100, holders=100, covered=95)
DOES_NOT_INHERIT = audit.Measurement(pairs=100, holders=100, covered=5)


@pytest.mark.parametrize(
    ("forward", "reverse", "expected"),
    [
        (INHERITS, DOES_NOT_INHERIT, audit.Proposal.GENERALISES),
        (INHERITS, None, audit.Proposal.GENERALISES),
        (INHERITS, audit.Measurement(pairs=3, holders=3, covered=3), audit.Proposal.GENERALISES),
        (INHERITS, INHERITS, audit.Proposal.DOES_NOT),
        (DOES_NOT_INHERIT, DOES_NOT_INHERIT, audit.Proposal.DOES_NOT),
        (audit.Measurement(pairs=5, holders=5, covered=5), None, audit.Proposal.DOES_NOT),
        (audit.Measurement(0, 0, 0), None, audit.Proposal.DOES_NOT),
        (audit.TimedOut(), None, audit.Proposal.UNMEASURED),
        (INHERITS, audit.TimedOut(), audit.Proposal.UNMEASURED),
    ],
)
def test_a_proposal_needs_inheritance_without_the_reverse_and_never_guesses_a_timeout(
    forward, reverse, expected
):
    assert audit.propose(forward, reverse, SETTINGS) is expected


def test_the_sensitivity_counts_the_relationships_whose_proposal_moves_with_a_threshold():
    measured = {
        "A": audit.Measurement(pairs=100, holders=100, covered=75),
        "A rev": DOES_NOT_INHERIT,
        "B": INHERITS,
        "B rev": INHERITS,
        "C": audit.TimedOut(),
    }
    reverse_of = {"A": "A rev", "B": "B rev"}

    result = audit.sensitivity(measured, reverse_of, ["A", "B", "C"], SETTINGS)
    by_setting = {(entry["threshold"], entry["value"]): entry for entry in result}

    assert by_setting[("min_share", 0.8)]["relationships"] == ["A"]
    assert by_setting[("min_share", 0.6)]["changed"] == 0
    assert by_setting[("max_reverse", 0.25)]["changed"] == 0
