import csv
import io
from importlib import resources


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


def test_predicate_subclasses_use_hyphen_minus():
    for filename in ("predicate_mapping.csv", "predicate_classification.csv"):
        assert all("–" not in row["subclass"] for row in _read_csv(filename))
