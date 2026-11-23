import pandas as pd
import pytest

from omop_graph.extensions.classification_data import (
    ClassificationDataError,
    check_mapping_against_cdm,
    hierarchy_direction,
)

NAMES = {"Maps to": "Maps to", "Is a": "Is a", "Subsumes": "Subsumes"}
REVERSE = {"Maps to": "Maps to", "Is a": "Subsumes", "Subsumes": "Is a"}
UP, DOWN, IDENTITY = (
    ("Hierarchy", "Taxonomic - up"),
    ("Hierarchy", "Taxonomic - down"),
    ("Identity", "x"),
)


def _mapping(**kinds: tuple[str | None, str | None]) -> pd.DataFrame:
    rows = [
        (rid.replace("_", " "), NAMES[rid.replace("_", " ")], *kind) for rid, kind in kinds.items()
    ]
    return pd.DataFrame(
        rows, columns=["relationship_id", "r_name", "predicate_kind", "predicate_subkind"]
    )


def _check(mapping: pd.DataFrame) -> None:
    check_mapping_against_cdm(mapping, NAMES, REVERSE)


GOOD = _mapping(Maps_to=IDENTITY, Is_a=UP, Subsumes=DOWN)


def test_a_consistent_mapping_passes():
    _check(GOOD)


def test_an_id_unknown_to_the_cdm_names_the_owner_of_its_name():
    mapping = GOOD.copy()
    mapping.loc[1, "relationship_id"] = "membership inference"

    with pytest.raises(
        ClassificationDataError, match="'membership inference'.*belongs to relationship_id 'Is a'"
    ):
        _check(mapping)


def test_a_name_differing_from_the_cdm_is_rejected():
    mapping = GOOD.copy()
    mapping.loc[1, "r_name"] = "Is a (SNOMED)"

    with pytest.raises(ClassificationDataError, match="relationship_name is 'Is a'"):
        _check(mapping)


def test_a_relationship_mapped_twice_is_rejected():
    with pytest.raises(ClassificationDataError, match="more than once.*Is a"):
        _check(pd.concat([GOOD, GOOD.iloc[[1]]]))


@pytest.mark.parametrize(
    "missing", [GOOD.iloc[:2], _mapping(Maps_to=IDENTITY, Is_a=UP, Subsumes=(None, None))]
)
def test_a_cdm_relationship_without_a_classification_is_rejected(missing):
    with pytest.raises(ClassificationDataError, match="no classification.*'Subsumes'"):
        _check(missing)


def test_the_reverse_of_a_hierarchy_relationship_must_be_hierarchy_with_the_opposite_direction():
    with pytest.raises(
        ClassificationDataError, match="'Is a' and its reverse 'Subsumes' are both 'up'"
    ):
        _check(_mapping(Maps_to=IDENTITY, Is_a=UP, Subsumes=UP))
    with pytest.raises(
        ClassificationDataError, match="'Is a' is Hierarchy but its reverse 'Subsumes' is Identity"
    ):
        _check(_mapping(Maps_to=IDENTITY, Is_a=UP, Subsumes=IDENTITY))


def test_hierarchy_direction_accepts_only_the_canonical_spelling():
    assert hierarchy_direction("Categorical - down") == "down"
    with pytest.raises(ClassificationDataError):
        hierarchy_direction("Taxonomic \u2013 up")


def test_an_empty_relationship_vocabulary_fails_clearly():
    with pytest.raises(RuntimeError, match="relationship vocabulary is empty"):
        check_mapping_against_cdm(GOOD, {}, {})
