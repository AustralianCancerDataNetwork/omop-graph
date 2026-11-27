import logging
import tempfile
from importlib import resources
from pathlib import Path
from typing import Annotated

import sqlalchemy as sa
import sqlalchemy.orm as so
import typer
from oa_configurator import ResolvedCDMDatabase, open_connection
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from orm_loader.backends import STAGING_SCHEMA, resolve_backend
from orm_loader.helpers.metadata import Base

from omop_graph.cli_utils import populate_test_data
from omop_graph.config import OmopGraphConfig
from omop_graph.db.session import open_cdm_sessions, resolve_cdm_database
from omop_graph.extensions.classification_data import (
    ClassificationDataError,
    check_mapping_against_cdm,
    read_classification_files,
)
from omop_graph.extensions.omop_alchemy import (
    RelationshipClass,
    RelationshipMapping,
    create_extension_tables,
)

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.callback()
def _main(
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase log verbosity (-v INFO, -vv DEBUG). Must come before the subcommand name.",
        ),
    ] = 0,
) -> None:
    OmopGraphConfig.configure_logging(verbosity=verbose)


@app.command()
def populate_with_test_data():
    """Populate the database with synthetic test data."""
    resolved = resolve_cdm_database()
    if not resolved.connection.test_only or not resolved.vocab_connection.test_only:
        raise RuntimeError(
            f"Refusing to populate {resolved.name!r} with synthetic test data: "
            "both its primary and vocab connections must be test_only."
        )
    with open_cdm_sessions(resolved) as cdm_sessions, cdm_sessions() as session:
        populate_test_data(session)


def packaged_predicate_csv_dir() -> Path:
    """Directory of the predicate CSVs shipped inside the package.

    These live under ``src/omop_graph/data`` rather than a top-level ``config/``
    so they reach the wheel; a top-level directory is outside
    ``[tool.hatch.build.targets.wheel] packages`` and never gets packaged,
    leaving ``relationship-classification`` unrunnable from a normal install.
    """
    return Path(str(resources.files("omop_graph") / "data"))


def relationship_classification(
    pred_class_dir: str | None = None,
    *,
    strict: bool = False,
    resolved: ResolvedCDMDatabase | None = None,
) -> None:
    """Load pre-classified predicates into the database.

    A CDM relationship without a classification, and a classified id that is not in the
    CDM, are logged as warnings and the unknown ids are not loaded.

    Parameters
    ----------
    pred_class_dir : str, optional
        Path to the directory containing `predicate_classification.csv` and
        `predicate_mapping.csv`. Defaults to the copies shipped with
        omop-graph.
    strict : bool, default False
        Raise on the two findings above instead of warning.
    resolved : ResolvedCDMDatabase, optional
        Database to load into. Defaults to the active config's CDM database.
        When the vocabulary has its own database, ``RelationshipMapping`` is
        created without its FK to ``relationship.relationship_id``.

    Raises
    ------
    ClassificationDataError
        If the files are inconsistent or do not match the CDM ``relationship`` table.
        The checks run before the existing tables are dropped.
    """
    if resolved is None:
        resolved = resolve_cdm_database()
    with open_cdm_sessions(resolved) as cdm_sessions:
        with cdm_sessions() as session:
            engine = session.get_bind(RelationshipMapping).engine
            relationships = session.execute(
                sa.select(
                    Relationship.relationship_id,
                    Relationship.relationship_name,
                    Relationship.reverse_relationship_id,
                )
            ).all()
        cdm_relationships = {rid: (name, reverse) for rid, name, reverse in relationships}
        _load_relationship_classification(
            pred_class_dir,
            resolved=resolved,
            engine=engine,
            relationships=cdm_relationships,
            strict=strict,
        )


def _load_relationship_classification(
    pred_class_dir: str | None,
    *,
    resolved: ResolvedCDMDatabase,
    engine: sa.Engine,
    relationships: dict[str, tuple[str, str]],
    strict: bool,
) -> None:
    """Body of :func:`relationship_classification`, on *engine*, which hosts the extension tables."""
    df_rel_cls_to_export, df_rel_mapping = read_classification_files(
        Path(pred_class_dir) if pred_class_dir else packaged_predicate_csv_dir()
    )
    report = check_mapping_against_cdm(df_rel_mapping, relationships, strict=strict)
    for warning in report.warnings:
        logger.warning(warning)
    df_rel_mapping_to_export = df_rel_mapping.loc[
        ~df_rel_mapping["relationship_id"].isin(report.unknown_ids),
        ["relationship_id", "predicate_kind", "predicate_subkind", "direction", "generalises"],
    ]

    loader_backend = resolve_backend(engine, staging_schema_tag=STAGING_SCHEMA)

    drop_staging_sql = (
        sa.text(
            "DROP TABLE IF EXISTS "
            f"{loader_backend.qualified_staging_name(RelationshipMapping.__tablename__)} CASCADE"
        ),
        sa.text(
            "DROP TABLE IF EXISTS "
            f"{loader_backend.qualified_staging_name(RelationshipClass.__tablename__)} CASCADE"
        ),
    )
    with open_connection(engine) as connection:
        for stmt in drop_staging_sql:
            connection.execute(stmt)

    # DROP TYPE IF EXISTS predicatekindenum was dead code: the Enum column
    # never set an explicit name=, so SQLAlchemy's generated type name is
    # actually "predicatekind". drop_all(tables=[...]) already drops the
    # shared Enum type exactly once, deduped, since both tables using it
    # are always in the same tables= list.
    tables_to_drop = [
        RelationshipMapping.__table__,
        RelationshipClass.__table__,
    ]
    with open_connection(engine) as connection:
        Base.metadata.drop_all(bind=connection, tables=tables_to_drop, checkfirst=True)  # ty: ignore[invalid-argument-type]
        create_extension_tables(connection, resolved=resolved)

    with tempfile.TemporaryDirectory() as tmp_dir, so.Session(engine) as session:
        for model, df in zip(
            [RelationshipClass, RelationshipMapping],
            [df_rel_cls_to_export, df_rel_mapping_to_export],
        ):
            csv_path = Path(tmp_dir) / f"{model.__tablename__}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(
                f"Temporarily saved {len(df)} records to `{csv_path}` for model `{model.__name__}` for loading."
            )

            model.load_file(  # ty: ignore[invalid-argument-type]
                session,
                csv_path,
                dedupe=True,
                merge_strategy="replace",
                staging_schema_tag=STAGING_SCHEMA,
                resolved=resolved,
            )
            session.commit()

        for model in (RelationshipClass, RelationshipMapping):
            model.validate_foreign_keys(  # ty: ignore[invalid-argument-type]
                session, staging_schema_tag=STAGING_SCHEMA, resolved=resolved
            )


@app.command(name="relationship-classification")
def relationship_classification_cmd(
    pred_class_dir: Annotated[
        str | None,
        typer.Option(
            help=(
                "Path to the directory containing `predicate_classification.csv` "
                "and `predicate_mapping.csv`. Defaults to the copies shipped with "
                "omop-graph; pass a directory to override them."
            )
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help=(
                "Fail on a CDM relationship without a classification and on a classified "
                "id that is not in the CDM, instead of warning."
            ),
        ),
    ] = False,
):
    """Load pre-classified predicates into the database."""
    try:
        relationship_classification(pred_class_dir, strict=strict)
    except ClassificationDataError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    app()
