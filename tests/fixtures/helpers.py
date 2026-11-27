"""Shared test-helper primitives for vocab-schema fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sqlalchemy as sa
from oa_configurator import ResolvedCDMDatabase, ResolvedDatabase
from omop_alchemy.cdm.model.vocabulary import Concept, Concept_Class, Domain, Vocabulary

ATHENA_RELATIONSHIPS = Path(__file__).parent / "athena_relationships.csv"
"""Relationship id, name and reverse id of the Athena release v5.0 30-AUG-24.

Columns 1, 2 and 5 of the ``RELATIONSHIP.csv`` of that release.
"""


def require_cdm(resolved: ResolvedDatabase | None) -> ResolvedCDMDatabase:
    """Narrow a resolved database to the CDM kind that the CDM session factory requires."""
    assert isinstance(resolved, ResolvedCDMDatabase)
    return resolved


def table_of(model: Any) -> sa.Table:
    """The Table a mapped class is bound to, narrowed from the declarative base's FromClause."""
    table = model.__table__
    assert isinstance(table, sa.Table)
    return table


# The minimal vocab bootstrap set every split/non-split fixture needs, in FK dependency order.
VOCAB_TABLES = tuple(table_of(model) for model in (Domain, Vocabulary, Concept_Class, Concept))
