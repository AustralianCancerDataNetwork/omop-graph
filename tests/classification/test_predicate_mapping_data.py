import csv

from fixtures.helpers import ATHENA_RELATIONSHIPS

from omop_graph.cli import packaged_predicate_csv_dir
from omop_graph.extensions.classification_data import (
    check_mapping_against_cdm,
    read_classification_files,
)
from omop_graph.extensions.omop_alchemy import RelationshipClass, RelationshipMapping, Subkinds


def test_packaged_files_are_consistent_ascii_and_fit_the_table_columns():
    classification, mapping = read_classification_files(packaged_predicate_csv_dir())

    for frame in (classification, mapping):
        names = [*frame["predicate_kind"], *frame["predicate_subkind"]]
        assert all(name.isascii() for name in names)
    for model, frame in ((RelationshipClass, classification), (RelationshipMapping, mapping)):
        for column in model.__table__.columns:
            length = getattr(column.type, "length", None)
            if length and column.name in frame:
                assert frame[column.name].str.len().max() <= length, column.name


def test_packaged_mapping_matches_the_athena_relationship_table():
    with ATHENA_RELATIONSHIPS.open(encoding="utf-8", newline="") as source:
        athena = list(csv.DictReader(source))
    _, mapping = read_classification_files(packaged_predicate_csv_dir())

    report = check_mapping_against_cdm(
        mapping,
        {
            row["relationship_id"]: (row["relationship_name"], row["reverse_relationship_id"])
            for row in athena
        },
        strict=True,
    )

    assert report.warnings == []


def test_every_subkind_the_code_names_is_in_the_packaged_classification():
    classification, mapping = read_classification_files(packaged_predicate_csv_dir())
    packaged = set(zip(classification["predicate_kind"], classification["predicate_subkind"]))
    used = set(zip(mapping["predicate_kind"], mapping["predicate_subkind"]))

    named = {(subkind.kind.value, subkind.name) for subkind in Subkinds.named()}

    assert len(named) == len(Subkinds.named()) == 6
    assert named <= packaged
    assert named <= used
