"""The generalises audit script measures inheritance through ``Is a`` and proposes flags."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import sqlalchemy as sa
from fixtures.cdm_builders import (
    classified,
    concept,
    edge,
    relationship,
    relationship_class,
    sqlite_cdm,
)
from omop_alchemy.cdm.model.vocabulary import Concept, Concept_Relationship, Relationship
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph

_SCRIPT = Path(__file__).parents[2] / "scripts" / "generalises_audit.py"
_spec = importlib.util.spec_from_file_location("generalises_audit", _SCRIPT)
assert _spec and _spec.loader
audit = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = audit
_spec.loader.exec_module(audit)

HAS_SITE, SITE_OF = "Has finding site", "Finding site of"
LUNG_CANCER, SMALL_CELL, ADENO, UNRELATED = 1, 2, 3, 4
LUNG, THORACIC_ORGAN, RIGHT_LUNG, LIVER = 10, 11, 12, 13


@pytest.fixture
def sessions() -> Iterator[sessionmaker[Session]]:
    """Lung cancer has a site, and its children carry it, specialise it, or do not hold it."""
    rows = [
        *(concept(concept_id) for concept_id in (LUNG_CANCER, SMALL_CELL, ADENO, UNRELATED)),
        *(concept(concept_id) for concept_id in (LUNG, THORACIC_ORGAN, RIGHT_LUNG, LIVER)),
        relationship("Is a", "Subsumes"),
        relationship(HAS_SITE, SITE_OF),
        relationship(SITE_OF, HAS_SITE),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic"),
        relationship_class(PredicateKind.ATTRIBUTE, "Site"),
        classified(
            "Is a",
            PredicateKind.HIERARCHY,
            "Taxonomic",
            generalises=True,
            direction=HierarchyDirection.UP,
        ),
        classified(HAS_SITE, PredicateKind.ATTRIBUTE, "Site", generalises=True),
        classified(SITE_OF, PredicateKind.ATTRIBUTE, "Site"),
        edge(SMALL_CELL, "Is a", LUNG_CANCER),
        edge(ADENO, "Is a", LUNG_CANCER),
        edge(UNRELATED, "Is a", LUNG_CANCER),
        edge(RIGHT_LUNG, "Is a", LUNG),
        edge(LUNG, "Is a", THORACIC_ORGAN),
        edge(LIVER, "Is a", LIVER, invalid_reason="D"),
        edge(LUNG_CANCER, HAS_SITE, LUNG),
        edge(SMALL_CELL, HAS_SITE, LUNG),
        edge(ADENO, HAS_SITE, RIGHT_LUNG),
        edge(UNRELATED, HAS_SITE, LIVER),
    ]
    models = (Concept, Relationship, Concept_Relationship, RelationshipClass, RelationshipMapping)
    with sqlite_cdm(models, rows) as sessions:
        yield sessions


@pytest.fixture
def session(sessions) -> Iterator[Session]:
    with sessions() as session:
        yield session


@pytest.fixture
def kg(sessions) -> KnowledgeGraph:
    return KnowledgeGraph(sessions)


def test_a_child_carries_the_parent_object_by_holding_it_or_something_below_it(session, kg):
    measurement = audit.measure(session, kg, HAS_SITE, sample_edges=10, sample_pairs=10)

    assert measurement == audit.Measurement(pairs=3, holders=3, covered=2)


def test_a_child_that_holds_nothing_counts_as_a_pair_and_not_as_a_holder(session, kg):
    session.execute(
        sa.delete(Concept_Relationship).where(
            Concept_Relationship.concept_id_1 == UNRELATED,
            Concept_Relationship.relationship_id == HAS_SITE,
        )
    )
    session.commit()

    measurement = audit.measure(session, kg, HAS_SITE, sample_edges=10, sample_pairs=10)

    assert measurement == audit.Measurement(pairs=3, holders=2, covered=2)


def test_an_invalid_edge_is_not_a_held_object(session, kg):
    session.add(edge(UNRELATED, HAS_SITE, LUNG, invalid_reason="D"))
    session.commit()

    measurement = audit.measure(session, kg, HAS_SITE, sample_edges=10, sample_pairs=10)

    assert measurement.covered == 2


def _cancelled(sqlstate: str):
    def raise_it(*args, **kwargs):
        error = type("Cancelled", (Exception,), {"sqlstate": sqlstate})("statement")
        raise DBAPIError("statement", {}, error)

    return raise_it


def test_only_a_statement_timeout_is_reported_as_timed_out(session, kg, monkeypatch):
    monkeypatch.setattr(audit, "measure", _cancelled(audit.QUERY_CANCELED))
    assert audit.measure_within(session, kg, HAS_SITE, 10, 10, 1) == audit.TimedOut()

    monkeypatch.setattr(audit, "measure", _cancelled("42P01"))
    with pytest.raises(DBAPIError):
        audit.measure_within(session, kg, HAS_SITE, 10, 10, 1)


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
