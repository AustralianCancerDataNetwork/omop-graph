"""Shared test-helper primitives for vocab-schema fixtures."""

from __future__ import annotations

from typing import cast

import sqlalchemy as sa
from omop_alchemy.cdm.model.vocabulary import (
    Concept,
    Concept_Class,
    Domain,
    Vocabulary
)

VOCAB_TABLES = cast(
    "tuple[sa.Table, ...]",
    (Domain.__table__, Vocabulary.__table__, Concept_Class.__table__, Concept.__table__),
)
"""Domain/Vocabulary/Concept_Class/Concept: the minimal vocab bootstrap set
every split/non-split fixture needs, in FK dependency order. A consumer
needing more tables (e.g. Relationship, Concept_Relationship) extends this
tuple rather than redefining its own copy of the shared core.
"""
