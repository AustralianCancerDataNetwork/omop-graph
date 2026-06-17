import logging
import tempfile
from pathlib import Path
from typing import Annotated

import pandas as pd
import sqlalchemy as sa
import typer
from sqlalchemy.orm import sessionmaker

from orm_loader.helpers import bulk_load_context
from orm_loader.helpers.metadata import Base
from orm_loader.loaders.loader_interface import PandasLoader

from omop_graph.config import OmopGraphConfig
from omop_graph.db.session import get_engine
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping
from omop_graph.cli_utils import populate_test_data

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
    engine = get_engine()
    Session = sessionmaker(bind=engine, future=True)
    populate_test_data(Session())


@app.command()
def relationship_classification(
    pred_class_dir: Annotated[
        str,
        typer.Option(
            help="Path to the directory containing `predicate_classification.csv` and `predicate_mapping.csv`."
        ),
    ],
):
    """Load pre-classified predicates into the database."""
    pred_class_dir_pl = Path(pred_class_dir)
    if not pred_class_dir_pl.is_dir():
        raise NotADirectoryError(f"{pred_class_dir} is not a valid directory.")

    pred_mapping_file = pred_class_dir_pl / "predicate_mapping.csv"
    if not pred_mapping_file.is_file():
        raise FileNotFoundError(
            f"`predicate_mapping.csv` not found in {pred_class_dir_pl}"
        )
    pred_class_file = pred_class_dir_pl / "predicate_classification.csv"
    if not pred_class_file.is_file():
        raise FileNotFoundError(
            f"`predicate_classification.csv` not found in {pred_class_dir_pl}"
        )

    df_class = pd.read_csv(pred_class_file)
    df_mapping = pd.read_csv(pred_mapping_file)

    # 1. RelationshipClass
    df_rel_cls = df_class.rename(
        columns={"class": "predicate_kind", "subclass": "predicate_subkind"}
    )

    check = df_rel_cls.groupby(["predicate_kind", "predicate_subkind"])[
        ["description", "semantics", "inference"]
    ].nunique(dropna=True)
    violations = check[(check > 1).any(axis=1)]
    if not violations.empty:  # type: ignore[union-attr]
        conflicting_data = df_rel_cls[
            df_rel_cls["predicate_subkind"].isin(violations.index)
        ].sort_values("predicate_subkind")  # type: ignore[union-attr, arg-type, call-overload]
        logger.error(
            f"Validation Failed! {len(violations)} predicate_subkinds have conflicting definitions: {conflicting_data}"
        )
        raise AttributeError("Validation not passed")
    df_rel_cls_to_export = df_rel_cls.groupby(
        ["predicate_kind", "predicate_subkind"], as_index=False
    ).first()

    # 2. RelationshipMapping
    df_rel_mapping = df_mapping.rename(
        columns={
            "class": "predicate_kind",
            "subclass": "predicate_subkind",
            "r_id": "relationship_id",
        }
    )
    df_rel_mapping = df_rel_mapping[
        ["relationship_id", "predicate_kind", "predicate_subkind"]
    ].dropna(subset=["predicate_kind", "predicate_subkind"], how="all")  # type: ignore[call-overload]
    invalid_mask = (
        df_rel_mapping[["predicate_kind", "predicate_subkind"]].isna().any(axis=1)
    )
    dropped_ids = df_rel_mapping.loc[invalid_mask, "relationship_id"].unique().tolist()

    if dropped_ids:
        logger.warning(
            f"Dropping {len(dropped_ids)} relationships due to missing parent or child class: {dropped_ids}"
        )
    df_rel_mapping = df_rel_mapping.dropna(
        subset=["predicate_kind", "predicate_subkind"], how="any"
    )
    df_rel_mapping_to_export = df_rel_mapping.drop_duplicates(
        subset=["relationship_id", "predicate_kind", "predicate_subkind"]
    )

    engine = get_engine()
    Session = sessionmaker(bind=engine, future=True)
    session = Session()

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                f"DROP TABLE IF EXISTS {RelationshipMapping.staging_tablename()} CASCADE"
            )
        )  # type: ignore
        conn.execute(
            sa.text(
                f"DROP TABLE IF EXISTS {RelationshipClass.staging_tablename()} CASCADE"
            )
        )  # type: ignore
        conn.execute(sa.text("DROP TYPE IF EXISTS predicatekindenum CASCADE;"))

    tables_to_drop = [
        RelationshipMapping.__table__,
        RelationshipClass.__table__,
    ]
    Base.metadata.drop_all(bind=engine, tables=tables_to_drop, checkfirst=True)  # type: ignore
    Base.metadata.create_all(bind=engine, tables=tables_to_drop)  # type: ignore

    with tempfile.TemporaryDirectory() as tmp_dir:
        for model, df in zip(
            [RelationshipClass, RelationshipMapping],
            [df_rel_cls_to_export, df_rel_mapping_to_export],
        ):
            csv_path = Path(tmp_dir) / f"{model.__tablename__}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(
                f"Temporarily saved {len(df)} records to `{csv_path}` for model `{model.__name__}` for loading."
            )

            with bulk_load_context(session):
                model.load_csv(  # type: ignore
                    session,
                    csv_path,
                    dedupe=True,
                    merge_strategy="replace",
                    loader=PandasLoader(),
                )
                session.commit()


if __name__ == "__main__":
    app()
