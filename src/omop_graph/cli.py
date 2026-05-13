import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from typing import Annotated, Optional, Protocol
import pandas as pd
import os
from pathlib import Path

from dotenv import load_dotenv
import typer
import logging
app = typer.Typer()
logger = logging.getLogger(__name__)

from orm_loader.helpers import create_db, bulk_load_context
from orm_loader.loaders.loader_interface import PandasLoader
from orm_loader.helpers.metadata import Base
from omop_alchemy.cdm.handlers import (
    install_fulltext_columns,
    populate_fulltext_columns,
)
from omop_alchemy.cdm.base import (
    CDMTableBase,
)

from omop_alchemy.cdm.model.vocabulary import (
    Domain,
    Vocabulary,
    Concept_Class,
    Relationship,
    Concept,
    Concept_Ancestor,
    Concept_Relationship,
    Concept_Synonym
)
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping
from omop_graph.oaklib_interface.omop_factory import build_engine_string
from omop_graph.config import ENV_OMOP_VOCABULARY_DIR
from omop_graph.cli_utils import (
    populate_test_data
)


ATHENA_INITIAL_LOAD = [
    Domain,
    Vocabulary,
    Concept_Class,
    Relationship,
    Concept
]


ATHENA_SUBSEQUENT_LOAD = [
    Concept_Ancestor,
    Concept_Relationship,
    Concept_Synonym,
]

ATHENA_RELATIONSHIP_CLASSIFICATION_LOAD = [
    RelationshipClass,
    RelationshipMapping
]

def configure_logging_level(verbosity: int, reduce_logging: bool = False) -> None:
    """Configure global logging."""
    level_map = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    log_level = level_map.get(min(verbosity, 2), logging.DEBUG)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    if reduce_logging:
        exempt_loggers = ("omop_graph", "omop_emb")

        class _NamespaceAllowlistFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return record.name.startswith(exempt_loggers)

        allowlist_filter = _NamespaceAllowlistFilter()

        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.addFilter(allowlist_filter)

        existing_loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger_instance in existing_loggers:
            if logger_instance.name.startswith(exempt_loggers):
                continue
            logger_instance.setLevel(logging.CRITICAL + 1)
            logger_instance.propagate = False


def _enable_fulltext_sidecars(engine: sa.Engine, regconfig: str) -> None:
    install_fulltext_columns(engine)
    populate_fulltext_columns(engine, regconfig=regconfig)



@app.command()
def omop_cdm(
    add_test_data: Annotated[bool, typer.Option(help="Whether to add synthetic test data after loading Athena data. Omit if not used.")] = False,
    chunk_size: Annotated[int, typer.Option(
        "--chunk-size", "-c", 
        help="Number of rows to process in each chunk when loading large tables with fallback pandas loader.")] = 5000,
    pred_class_dir: Annotated[Optional[str], typer.Option(help="Path to the directory containing `predicate_classification.csv` and `predicate_mapping.csv`.")] = None,
    fulltext: Annotated[bool, typer.Option("--fulltext/--no-fulltext", help="Install and populate PostgreSQL full-text sidecars after loading the vocabulary tables.")] = False,
    fulltext_regconfig: Annotated[str, typer.Option("--fulltext-regconfig", help="PostgreSQL text search configuration to use when populating the full-text sidecars.")] = "english",
    verbosity: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity (up to two levels)")] = 0,
):
    """
    Instantiate the database from scratch by loading the Athena vocabularies.
    IMPORTANT: This will wipe the entire existing database in the db container.
    """
    configure_logging_level(verbosity)
    load_dotenv()

    engine_string = build_engine_string()
    engine = sa.create_engine(engine_string, future=True, echo=False)

    # Drop all existing tables for a fresh bootstrap
    metadata = Base.metadata
    metadata.reflect(bind=engine)
    metadata.drop_all(engine)
    
    # Re-init tables
    create_db(engine)

    Session = sessionmaker(bind=engine, future=True)
    session = Session()

    loader = PandasLoader()

    vocab_dir_env_var = os.getenv(ENV_OMOP_VOCABULARY_DIR)
    if vocab_dir_env_var is None:
        raise RuntimeError(f"{ENV_OMOP_VOCABULARY_DIR} environment variable not set. Please set it in your .env file to point to the Athena vocabulary CSV files directory.")
    vocab_dir = Path(vocab_dir_env_var).resolve()
    assert vocab_dir.exists(), f"Source path {vocab_dir} does not exist"

    with bulk_load_context(session):
       for model in ATHENA_INITIAL_LOAD:
           model.load_csv(
               session,
               vocab_dir / f"{model.__tablename__.upper()}.csv",
               dedupe=True,
               merge_strategy="upsert",
               loader=loader
           )
           logger.info(f"Loaded {model.__tablename__} table with Athena data.")
       session.commit()

    with bulk_load_context(session):
        for model in ATHENA_SUBSEQUENT_LOAD:
            model.load_csv(
                session,
                vocab_dir / f"{model.__tablename__.upper()}.csv",
                dedupe=True,
                chunksize=chunk_size,
                merge_strategy="replace",
                loader=loader
            )
            logger.info(f"Loaded {model.__tablename__} table with Athena data.")
            session.commit()

    if fulltext:
        try:
            _enable_fulltext_sidecars(engine, fulltext_regconfig)
            logger.info("Successfully enabled PostgreSQL full-text sidecars.")
        except Exception as exc:
            logger.error(f"Failed to enable PostgreSQL full-text sidecars: {exc}")
            logger.info("Continuing with bootstrap without full-text sidecars. You can rerun `omop-maint fulltext install` and `omop-maint fulltext populate` later.")

    try:
       relationship_classification(pred_class_dir)
       logger.info("Successfully ingested relationship classifications.")
    except Exception as e:
       logger.error(f"Failed to ingest predicate classifications: {e}")
       logger.info("Continuing with bootstrap without predicate classifications. Re-run cli `relationship-classification` command once the issue is resolved.")

    if add_test_data:
        populate_test_data(session)

@app.command()
def relationship_classification(
    pred_class_dir: Annotated[Optional[str], typer.Option(help="Path to the directory containing `predicate_classification.csv` and `predicate_mapping.csv`.")] = None,
    env_file: Annotated[Optional[str], typer.Option("--env-file", "-e", help="Path to the .env file containing database connection variables. If not provided, will look for .env in the current working directory.")] = None,
    verbosity: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity (up to two levels)")] = 0,
):
    """
    Method to get the pre-classified predicates into the database.
    """
    configure_logging_level(verbosity)
    load_dotenv(env_file)

    if pred_class_dir is None:
        pred_class_dir = str((Path(__file__).parent.parent.parent / "docs").resolve())

    pred_class_dir_pl = Path(pred_class_dir)

    pred_mapping_file = pred_class_dir_pl / "predicate_mapping.csv"
    if not pred_mapping_file.is_file():
        raise FileNotFoundError(f"`predicate_mapping.csv` not found in {pred_class_dir_pl}")
    pred_class_file = pred_class_dir_pl / "predicate_classification.csv"
    if not pred_class_file:
        raise FileNotFoundError(f"`predicate_classification.csv` not found in {pred_class_dir_pl}")

    df_class = pd.read_csv(pred_class_file)
    df_mapping = pd.read_csv(pred_mapping_file)
    

    # 1. RelationshipClass
    df_rel_cls = df_class.rename(columns={"class": "class_id", "subclass": "subclass_id"})

    # Only allow that a subclass_id maps exactly to one semantic and inference description
    check = df_rel_cls.groupby(["class_id", "subclass_id"])[["description", "semantics", "inference"]].nunique(dropna=True)
    violations = check[(check > 1).any(axis=1)]
    if not violations.empty:
        conflicting_data = df_rel_cls[df_rel_cls["subclass_id"].isin(violations.index)].sort_values("subclass_id")
        logger.error(f"Validation Failed! {len(violations)} subclass_ids have conflicting definitions: {conflicting_data}")        
        raise AttributeError("Validation not passed")
    df_rel_cls_to_export = df_rel_cls.groupby(["class_id", "subclass_id"], as_index=False).first()

    # 2. RelationshipMapping
    df_rel_mapping = df_mapping.rename(columns={"class": "class_id", "subclass": "subclass_id", "r_id": "relationship_id"})
    # Same order as relationship_class.py
    df_rel_mapping = df_rel_mapping[["relationship_id", "class_id", "subclass_id"]].dropna(subset=['class_id', 'subclass_id'], how='all')
    invalid_mask = df_rel_mapping[['class_id', 'subclass_id']].isna().any(axis=1)
    dropped_ids = df_rel_mapping.loc[invalid_mask, 'relationship_id'].unique().tolist()

    if dropped_ids:
        logger.warning(f"Dropping {len(dropped_ids)} relationships due to missing parent or child class: {dropped_ids}")
    df_rel_mapping = df_rel_mapping.dropna(subset=['class_id', 'subclass_id'], how='any')
    df_rel_mapping_to_export = df_rel_mapping.drop_duplicates(subset=["relationship_id", "class_id", "subclass_id"])

    # Save and then load again
    vocab_dir_env_var = os.getenv(ENV_OMOP_VOCABULARY_DIR)
    if vocab_dir_env_var is None:
        raise RuntimeError(f"{ENV_OMOP_VOCABULARY_DIR} environment variable not set. Please set it in your .env file to point to the Athena vocabulary CSV files directory.")
    vocab_dir = Path(vocab_dir_env_var).resolve()
    assert vocab_dir.exists(), f"Source path {vocab_dir} does not exist"

    engine_string = build_engine_string()
    engine = sa.create_engine(engine_string, future=True, echo=False)
    Session = sessionmaker(bind=engine, future=True)
    session = Session()

    # Drop the tables
    with engine.begin() as conn:
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {RelationshipMapping.staging_tablename()} CASCADE"))  # type: ignore
        conn.execute(sa.text(f"DROP TABLE IF EXISTS {RelationshipClass.staging_tablename()} CASCADE"))  # type: ignore
        conn.execute(sa.text("DROP TYPE IF EXISTS classidenum CASCADE;"))

    tables_to_drop = [
        RelationshipMapping.__table__, 
        RelationshipClass.__table__
    ]
    Base.metadata.drop_all(bind=engine, tables=tables_to_drop, checkfirst=True)  # type: ignore
    Base.metadata.create_all(bind=engine, tables=tables_to_drop)  # type: ignore

    for model, df in zip([RelationshipClass, RelationshipMapping], [df_rel_cls_to_export, df_rel_mapping_to_export]):
        csv_path = vocab_dir / f"{model.__tablename__.upper()}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df)} records to `{csv_path}` for model `{model.__name__}`")

        with bulk_load_context(session):
            model.load_csv(  # type: ignore
                session,
                csv_path,
                dedupe=True,
                merge_strategy="replace",
                loader=PandasLoader()
            )
            session.commit()

if __name__ == "__main__":
    app()