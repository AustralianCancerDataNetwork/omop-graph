import pytest

from omop_alchemy.cdm.handlers.fulltext import (
    CONCEPT_NAME_TSVECTOR_COLUMN,
    CONCEPT_SYNONYM_NAME_TSVECTOR_COLUMN,
    register_optional_fulltext_columns,
    unregister_optional_fulltext_columns,
)
from omop_alchemy.cdm.model.vocabulary import Concept, Concept_Synonym

from omop_graph.extensions.emb import MissingExtensionError
from omop_graph.graph.queries import q_concept_name_fulltext


@pytest.mark.parametrize("synonym", [False, True])
def test_fulltext_query_requires_registered_tsvector_columns(synonym: bool):
    """Full-text queries fail cleanly when optional tsvector metadata is absent."""
    had_name_column = CONCEPT_NAME_TSVECTOR_COLUMN in Concept.__table__.c
    had_synonym_column = CONCEPT_SYNONYM_NAME_TSVECTOR_COLUMN in Concept_Synonym.__table__.c

    unregister_optional_fulltext_columns()
    try:
        with pytest.raises(MissingExtensionError):
            q_concept_name_fulltext("kidney cancer", synonym=synonym)
    finally:
        if had_name_column or had_synonym_column:
            register_optional_fulltext_columns()