import pandas as pd
import pytest

from omop_graph.extensions.classification_data import (
    ClassificationDataError,
    check_classification_files,
    check_mapping_against_cdm,
)

CDM = {
    "Maps to": "Non-standard to Standard map (OMOP)",
    "Is a": "Is a",
    "Has Category": "Has Category",
}


def _classification(*pairs: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["predicate_kind", "predicate_subkind"])


def _mapping(*rows: tuple[str, str, str | None, str | None]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["relationship_id", "r_name", "predicate_kind", "predicate_subkind"]
    )


COMPLETE = _mapping(
    ("Maps to", "Non-standard to Standard map (OMOP)", "Identity", "Exact equivalence"),
    ("Is a", "Is a", "Hierarchy", "Taxonomic - up"),
    ("Has Category", "Has Category", "Hierarchy", "Categorical - up"),
)


def test_consistent_files_and_mapping_pass():
    classification = _classification(
        ("Identity", "Exact equivalence"),
        ("Hierarchy", "Taxonomic - up"),
        ("Hierarchy", "Categorical - up"),
    )

    check_classification_files(classification, COMPLETE)
    check_mapping_against_cdm(COMPLETE, CDM)


def test_files_reject_non_ascii_subkinds():
    classification = _classification(("Hierarchy", "Taxonomic \u2013 up"))
    mapping = _mapping(("Is a", "Is a", "Hierarchy", "Taxonomic \u2013 up"))

    with pytest.raises(ClassificationDataError, match="non-ASCII"):
        check_classification_files(classification, mapping)


def test_files_reject_a_relationship_mapped_twice():
    classification = _classification(("Hierarchy", "Taxonomic - up"), ("Identity", "mapping"))
    mapping = _mapping(
        ("Is a", "Is a", "Hierarchy", "Taxonomic - up"),
        ("Is a", "Is a", "Identity", "mapping"),
    )

    with pytest.raises(ClassificationDataError, match="more than once.*Is a"):
        check_classification_files(classification, mapping)


def test_files_reject_a_subkind_missing_from_the_class_file():
    classification = _classification(("Hierarchy", "Taxonomic - up"))
    mapping = _mapping(("Is a", "Is a", "Hierarchy", "Taxonomic - down"))

    with pytest.raises(ClassificationDataError, match="missing from predicate_classification"):
        check_classification_files(classification, mapping)


def test_files_report_every_problem_at_once():
    classification = _classification(("Hierarchy", "Taxonomic \u2013 up"))
    mapping = _mapping(
        ("Is a", "Is a", "Hierarchy", "Taxonomic \u2013 up"),
        ("Is a", "Is a", "Hierarchy", "Taxonomic \u2013 up"),
    )

    with pytest.raises(ClassificationDataError) as error:
        check_classification_files(classification, mapping)

    assert "non-ASCII" in str(error.value)
    assert "more than once" in str(error.value)


def test_mapping_with_an_id_unknown_to_the_cdm_names_the_owner_of_its_name():
    mapping = pd.concat(
        [
            COMPLETE.iloc[:2],
            _mapping(("membership inference", "Has Category", "Hierarchy", "Categorical - up")),
        ]
    )

    with pytest.raises(
        ClassificationDataError,
        match="'membership inference'.*belongs to relationship_id 'Has Category'",
    ):
        check_mapping_against_cdm(mapping, CDM)


def test_mapping_with_a_name_differing_from_the_cdm_is_rejected():
    mapping = COMPLETE.copy()
    mapping.loc[1, "r_name"] = "Is a (SNOMED)"

    with pytest.raises(ClassificationDataError, match="relationship_name is 'Is a'"):
        check_mapping_against_cdm(mapping, CDM)


def test_mapping_without_a_name_column_is_checked_by_id_only():
    check_mapping_against_cdm(COMPLETE.drop(columns="r_name"), CDM)

    with pytest.raises(ClassificationDataError, match="'Nope'"):
        check_mapping_against_cdm(
            pd.concat([COMPLETE, _mapping(("Nope", "x", "Identity", "x"))]), CDM
        )


def test_a_cdm_relationship_without_a_classification_is_rejected():
    with pytest.raises(ClassificationDataError, match="1 CDM relationships.*'Has Category'"):
        check_mapping_against_cdm(COMPLETE.iloc[:2], CDM)


def test_a_mapping_row_without_a_class_does_not_classify_its_relationship():
    mapping = COMPLETE.copy()
    mapping.loc[2, ["predicate_kind", "predicate_subkind"]] = None

    with pytest.raises(ClassificationDataError, match="'Has Category'"):
        check_mapping_against_cdm(mapping, CDM)


def test_empty_relationship_vocabulary_fails_clearly():
    with pytest.raises(RuntimeError, match="relationship vocabulary is empty"):
        check_mapping_against_cdm(COMPLETE, {})
