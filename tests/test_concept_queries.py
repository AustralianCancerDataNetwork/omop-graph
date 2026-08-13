"""Integration tests for OMOP Alchemy's canonical concept-query semantics."""

from __future__ import annotations

from datetime import date

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from omop_alchemy.cdm.model.vocabulary import Concept
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.graph.nodes import ConceptView
from omop_graph.graph.queries import (
    q_concept_filtered,
    q_concept_name_match,
    q_concept_views,
    q_entities,
)


@pytest.fixture()
def concept_engine() -> sa.Engine:
    engine = sa.create_engine("sqlite+pysqlite:///:memory:", future=True)
    Concept.__table__.create(engine)

    valid_from = date(2000, 1, 1)
    valid_until = date(2099, 12, 31)

    def concept(
        concept_id: int,
        *,
        standard_concept: str | None,
        invalid_reason: str | None,
    ) -> Concept:
        return Concept(
            concept_id=concept_id,
            concept_name="Shared label",
            domain_id="Condition",
            vocabulary_id="SNOMED",
            concept_class_id="Clinical Finding",
            standard_concept=standard_concept,
            concept_code=f"TEST-{concept_id}",
            valid_start_date=valid_from,
            valid_end_date=valid_until,
            invalid_reason=invalid_reason,
        )

    with Session(engine) as session:
        session.add_all(
            [
                concept(1, standard_concept="S", invalid_reason=None),
                concept(2, standard_concept="C", invalid_reason=" "),
                concept(3, standard_concept=None, invalid_reason=None),
                concept(4, standard_concept="S", invalid_reason="U"),
                concept(5, standard_concept=" ", invalid_reason="X"),
            ]
        )
        session.commit()

    return engine


def test_concept_filter_applies_canonical_graph_constraints(
    concept_engine: sa.Engine,
) -> None:
    constraint = ConceptFilter(
        domains=("Condition",),
        require_standard=True,
        require_active=True,
    )

    with Session(concept_engine) as session:
        concept_ids = tuple(
            session.scalars(constraint.apply(sa.select(Concept.concept_id)))
        )

    assert concept_ids == (1, 2)


def test_concept_filter_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ConceptFilter(limit=0)


def test_concept_views_use_canonical_standard_and_active_flags(
    concept_engine: sa.Engine,
) -> None:
    with Session(concept_engine) as session:
        views = {
            view.concept_id: view
            for view in (
                ConceptView.from_row(row)
                for row in session.execute(
                    q_concept_views((1, 2, 3, 4, 5), sort=False)
                )
            )
        }

    assert views[1].standard_concept is True
    assert views[2].standard_concept is True  # classification concepts are standard
    assert views[3].standard_concept is False
    assert views[5].standard_concept is False

    assert views[1].is_active is True
    assert views[2].is_active is True  # blank/whitespace invalid_reason is treated as unset
    assert views[4].is_active is False
    assert views[5].is_active is False


def test_label_query_projects_and_orders_canonical_flags(
    concept_engine: sa.Engine,
) -> None:
    with Session(concept_engine) as session:
        rows = session.execute(q_concept_name_match("Shared label")).all()

    assert [row.concept_id for row in rows] == [1, 2, 4, 3, 5]
    statuses = {
        row.concept_id: (row.is_standard, row.is_active) for row in rows
    }
    assert statuses == {
        1: (True, True),
        2: (True, True),
        3: (False, True),
        4: (True, False),
        5: (False, False),
    }


def test_graph_listing_queries_delegate_to_shared_filter(
    concept_engine: sa.Engine,
) -> None:
    with Session(concept_engine) as session:
        active_standard_ids = tuple(session.scalars(q_entities(domain="Condition")))
        all_standard_ids = tuple(
            session.scalars(
                q_concept_filtered(
                    domain_id="Condition",
                    vocabulary_id="SNOMED",
                )
            )
        )

    assert active_standard_ids == (1, 2)
    assert all_standard_ids == (1, 2, 4)
