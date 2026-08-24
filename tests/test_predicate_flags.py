"""Relationship varchar-boolean flags must not be read by truthiness.

``relationship.is_hierarchical`` and ``defines_ancestry`` are OMOP ``VARCHAR(1)``
columns holding the *strings* ``'1'`` and ``'0'``. ``'0'`` is truthy in Python,
so a bare ``bool()`` over the raw column reports every relationship as
hierarchical. omop-alchemy >= 1.1 exposes predicates that convert in SQL; the
query layer projects those, so no consumer here ever sees the raw string.
"""

from __future__ import annotations

from omop_graph.graph.kg import KnowledgeGraph


def test_zero_flag_is_not_read_as_true(mock_cdm_kg: KnowledgeGraph) -> None:
    """The fixture's relationships are all ``is_hierarchical='0'``.

    Under the previous ``bool(row.is_hierarchical)`` conversion these came back
    True, because ``bool('0')`` is True.
    """
    predicate = mock_cdm_kg.predicate("maps to")

    assert predicate.is_hierarchical is False
    assert predicate.anc_down is False
    assert predicate.anc_up is False


def test_single_and_bulk_lookups_agree(mock_cdm_kg: KnowledgeGraph) -> None:
    """predicate() and predicates() converted the same column two
    different ways, so they disagreed for every ``'0'`` relationship."""
    bulk = {p.relationship_id: p for p in mock_cdm_kg.predicates()}

    assert bulk  # sanity: the fixture defines relationships

    for relationship_id, bulk_predicate in bulk.items():
        single = mock_cdm_kg.predicate(relationship_id)
        assert single.is_hierarchical is bulk_predicate.is_hierarchical, relationship_id
        assert single.anc_down is bulk_predicate.anc_down, relationship_id
        assert single.anc_up is bulk_predicate.anc_up, relationship_id


def test_flags_are_real_booleans(mock_cdm_kg: KnowledgeGraph) -> None:
    """Converted in SQL, so consumers get bool rather than '1'/'0' strings."""
    for predicate in mock_cdm_kg.predicates():
        assert isinstance(predicate.is_hierarchical, bool)
        assert isinstance(predicate.anc_down, bool)
        assert isinstance(predicate.anc_up, bool)
