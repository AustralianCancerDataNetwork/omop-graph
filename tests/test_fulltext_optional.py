import pytest

from sqlalchemy import Engine
from omop_alchemy.backends import FullTextError

from omop_graph.graph.queries import q_concept_name_fulltext


@pytest.mark.parametrize("synonym", [False, True])
def test_fulltext_query_requires_tsvector_columns(synonym: bool, mock_cdm_engine: Engine):
    """Full-text query raises FullTextError when tsvector columns are absent from the database.

    The mock CDM engine is SQLite and never has tsvector columns, so the guard in
    q_concept_name_fulltext (which inspects the live DB schema) always fires here.
    """
    with pytest.raises(FullTextError):
        q_concept_name_fulltext("kidney cancer", synonym=synonym, engine=mock_cdm_engine)
