import csv
import io
from importlib import resources
from pathlib import Path

import pandas as pd

from omop_graph.extensions.classification_data import check_mapping_against_cdm

ATHENA_RELATIONSHIPS = Path(__file__).parent / "fixtures" / "athena_relationships.csv"
"""Relationship id, name and reverse id of the Athena release v5.0 30-AUG-24."""


def _read_csv(filename: str) -> list[dict[str, str]]:
    path = resources.files("omop_graph").joinpath("data", filename)
    with path.open("rb") as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def test_every_mapping_class_subclass_has_a_classification():
    mapping_pairs = {(row["class"], row["subclass"]) for row in _read_csv("predicate_mapping.csv")}
    classification_pairs = {
        (row["class"], row["subclass"]) for row in _read_csv("predicate_classification.csv")
    }

    assert mapping_pairs <= classification_pairs


def test_predicate_classes_and_subclasses_are_ascii():
    for filename in ("predicate_mapping.csv", "predicate_classification.csv"):
        assert all(
            row["class"].isascii() and row["subclass"].isascii() for row in _read_csv(filename)
        )


def test_packaged_mapping_matches_the_athena_relationship_table():
    with ATHENA_RELATIONSHIPS.open(encoding="utf-8", newline="") as source:
        athena = list(csv.DictReader(source))
    mapping = pd.DataFrame(_read_csv("predicate_mapping.csv")).rename(
        columns={
            "class": "predicate_kind",
            "subclass": "predicate_subkind",
            "r_id": "relationship_id",
        }
    )

    check_mapping_against_cdm(
        mapping,
        {row["relationship_id"]: row["relationship_name"] for row in athena},
        {row["relationship_id"]: row["reverse_relationship_id"] for row in athena},
    )
