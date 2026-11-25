from __future__ import annotations

from datetime import date
from typing import cast

import pytest
import sqlalchemy as sa
from omop_alchemy.cdm.model.vocabulary import Concept_Relationship
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from omop_graph.extensions.omop_alchemy import PredicateKind, PredicateSubkind
from omop_graph.graph.edges import EdgeFilter
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.queries import q_edges, q_relationships

SOURCE = 900001
IDENTITY_MAPPING = PredicateSubkind(PredicateKind.IDENTITY, "mapping")


def _sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def test_kinds_and_subkinds_must_be_the_typed_values() -> None:
    with pytest.raises(TypeError, match="PredicateKind"):
        EdgeFilter(predicate_kinds=frozenset({"Identity"}))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="PredicateSubkind"):
        EdgeFilter(predicate_subkinds=frozenset({"mapping"}))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("edge_filter", "expected"),
    [
        (EdgeFilter(predicate_subkinds=frozenset({IDENTITY_MAPPING})), 1),
        (EdgeFilter(predicate_ids=frozenset({"maps to"})), 1),
        (EdgeFilter(predicate_ids=frozenset({"mapped from"})), 0),
        (
            EdgeFilter(
                predicate_ids=frozenset({"maps to"}),
                predicate_kinds=frozenset({PredicateKind.IDENTITY}),
            ),
            1,
        ),
        (
            EdgeFilter(
                predicate_ids=frozenset({"maps to"}),
                predicate_kinds=frozenset({PredicateKind.HIERARCHY}),
            ),
            0,
        ),
    ],
)
def test_filter_fields_combine_with_and(
    mock_cdm_kg: KnowledgeGraph, edge_filter: EdgeFilter, expected: int
) -> None:
    edges = mock_cdm_kg.edges(SOURCE, direction="out", edge_filter=edge_filter)
    triples = tuple(mock_cdm_kg.relationships((SOURCE,), edge_filter=edge_filter))

    assert len(edges) == len(triples) == expected


def test_an_empty_predicate_set_selects_no_edge_where_none_selects_all(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    nothing = EdgeFilter(predicate_ids=frozenset())

    assert len(mock_cdm_kg.edges(SOURCE, direction="out", edge_filter=nothing)) == 0
    assert tuple(mock_cdm_kg.relationships((SOURCE,), edge_filter=nothing)) == ()
    assert mock_cdm_kg.closure([SOURCE], edge_filter=nothing) == {SOURCE: frozenset()}
    assert len(mock_cdm_kg.edges(SOURCE, direction="out", edge_filter=EdgeFilter())) > 0


def test_an_unknown_subkind_names_the_valid_ones(mock_cdm_kg: KnowledgeGraph) -> None:
    typo = EdgeFilter(
        predicate_subkinds=frozenset({PredicateSubkind(PredicateKind.IDENTITY, "mappin")})
    )

    with pytest.raises(ValueError, match="mappin.*mapping"):
        mock_cdm_kg.edges(SOURCE, direction="out", edge_filter=typo)


def test_the_same_subkind_name_under_another_kind_is_unknown(mock_cdm_kg: KnowledgeGraph) -> None:
    wrong_kind = EdgeFilter(
        predicate_subkinds=frozenset({PredicateSubkind(PredicateKind.HIERARCHY, "mapping")})
    )

    with pytest.raises(ValueError, match="Unknown predicate subkinds"):
        mock_cdm_kg.edges(SOURCE, direction="out", edge_filter=wrong_kind)


@pytest.mark.parametrize(
    "build", [lambda f: q_edges(1, "out", f), lambda f: q_relationships(None, None, f)]
)
def test_edges_and_relationships_share_the_validity_and_domain_conditions(build) -> None:
    default_sql = _sql(build(EdgeFilter()))
    everything_sql = _sql(build(EdgeFilter(active_only=False)))
    within_domain_sql = _sql(build(EdgeFilter(within_domain=True)))

    assert "invalid_reason" in default_sql
    assert "invalid_reason" not in everything_sql.split("FROM")[1]
    assert "domain_id" not in default_sql
    assert "domain_id" in within_domain_sql


@pytest.mark.parametrize(
    ("edge_filter", "expected_objects"),
    [
        (EdgeFilter(active_only=False), {2, 3}),
        (EdgeFilter(), {3}),
        (EdgeFilter(active_only=False, on=date(2015, 1, 1)), {2, 3}),
        (EdgeFilter(active_only=False, on=date(2025, 1, 1)), {3}),
        (EdgeFilter(on=date(2015, 1, 1)), {3}),
    ],
)
def test_on_selects_edges_by_validity_period_independently_of_active_only(
    edge_filter: EdgeFilter, expected_objects: set[int]
) -> None:
    engine = sa.create_engine("sqlite+pysqlite:///:memory:").execution_options(
        schema_translate_map={"vocab": None}
    )
    cast(sa.Table, Concept_Relationship.__table__).create(engine)
    with Session(engine) as session:
        session.add_all(
            [
                Concept_Relationship(
                    concept_id_1=1,
                    concept_id_2=2,
                    relationship_id="Is a",
                    valid_start_date=date(2010, 1, 1),
                    valid_end_date=date(2020, 1, 1),
                    invalid_reason="D",
                ),
                Concept_Relationship(
                    concept_id_1=1,
                    concept_id_2=3,
                    relationship_id="Is a",
                    valid_start_date=date(2010, 1, 1),
                    valid_end_date=date(2099, 12, 31),
                    invalid_reason=None,
                ),
            ]
        )
        session.commit()
        rows = session.execute(q_edges(1, "out", edge_filter))
        objects = {row.object_id for row in rows}
    engine.dispose()

    assert objects == expected_objects
