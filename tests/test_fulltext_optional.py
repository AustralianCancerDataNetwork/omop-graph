import pytest

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker
from omop_alchemy.backends import FullTextError

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.graph.queries import q_concept_name_fulltext


@pytest.mark.parametrize("synonym", [False, True])
@pytest.mark.postgresql
@pytest.mark.db_dialect
def test_fulltext_query_requires_tsvector_columns(
    synonym: bool, mock_cdm_engine: Engine
):
    """Full-text query raises FullTextError when tsvector columns are absent from the database.

    The mock CDM's own table creation never adds tsvector columns (that's a
    separate, opt-in fulltext-install step), so the guard in
    q_concept_name_fulltext (which inspects the live DB schema) always fires here.
    """
    with pytest.raises(FullTextError):
        q_concept_name_fulltext(
            "kidney cancer", synonym=synonym, engine=mock_cdm_engine
        )


def test_knowledge_graph_fulltext_lookup_skips_unsupported_sqlite(
    mock_cdm_engine_sqlite: Engine,
):
    kg = KnowledgeGraph(sessionmaker(bind=mock_cdm_engine_sqlite))
    assert kg.concept_lookup("kidney cancer", LabelMatchKind.FTS) == ()


def test_knowledge_graph_rejects_engine_instead_of_session_factory():
    engine = sa.create_engine("sqlite+pysqlite:///:memory:")
    try:
        with pytest.raises(TypeError, match="callable session factory.*not an Engine"):
            KnowledgeGraph(engine)
    finally:
        engine.dispose()
