import pandas as pd
import pytest

from omop_graph.extensions.classification_data import (
    ClassificationDataError,
    check_classification,
    check_mapping_against_cdm,
)
from omop_graph.extensions.omop_alchemy import HierarchyDirection, PredicateKind

RELATIONSHIPS = {
    "Maps to": ("Maps to", "Maps to"),
    "Is a": ("Is a", "Subsumes"),
    "Subsumes": ("Subsumes", "Is a"),
}
HIERARCHY, IDENTITY_KIND = PredicateKind.HIERARCHY.value, PredicateKind.IDENTITY.value
UP, DOWN, IDENTITY = (
    (HIERARCHY, "Taxonomic - up", HierarchyDirection.UP.value),
    (HIERARCHY, "Taxonomic - down", HierarchyDirection.DOWN.value),
    (IDENTITY_KIND, "x", None),
)


def _mapping(*rows: tuple[str, tuple[str | None, str | None, str | None]]) -> pd.DataFrame:
    rows = [(rid, RELATIONSHIPS[rid][0], *kind) for rid, kind in rows]
    return pd.DataFrame(
        rows,
        columns=["relationship_id", "r_name", "predicate_kind", "predicate_subkind", "direction"],
    )


def _check(mapping: pd.DataFrame) -> None:
    check_mapping_against_cdm(mapping, RELATIONSHIPS)


GOOD = _mapping(("Maps to", IDENTITY), ("Is a", UP), ("Subsumes", DOWN))


def test_a_consistent_mapping_passes():
    _check(GOOD)


def test_an_id_unknown_to_the_cdm_is_rejected():
    mapping = GOOD.copy()
    mapping.loc[1, "relationship_id"] = "membership inference"

    with pytest.raises(ClassificationDataError, match="'membership inference'.*not in the CDM"):
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
    "missing",
    [
        GOOD.iloc[:2],
        _mapping(("Maps to", IDENTITY), ("Is a", UP), ("Subsumes", (None, None, None))),
    ],
)
def test_a_cdm_relationship_without_a_classification_is_rejected(missing):
    with pytest.raises(ClassificationDataError, match="no classification.*'Subsumes'"):
        _check(missing)


def test_the_reverse_of_a_hierarchy_relationship_must_be_hierarchy_with_the_opposite_direction():
    with pytest.raises(
        ClassificationDataError, match="'Is a' and its reverse 'Subsumes' are both 'up'"
    ):
        _check(_mapping(("Maps to", IDENTITY), ("Is a", UP), ("Subsumes", UP)))
    with pytest.raises(
        ClassificationDataError, match="'Is a' is Hierarchy but its reverse 'Subsumes' is Identity"
    ):
        _check(_mapping(("Maps to", IDENTITY), ("Is a", UP), ("Subsumes", IDENTITY)))


@pytest.mark.parametrize(
    ("kind", "direction", "message"),
    [
        (HIERARCHY, None, "without an up or down direction.*'x'"),
        (HIERARCHY, "sideways", "without an up or down direction.*'x'"),
        (IDENTITY_KIND, HierarchyDirection.UP.value, "not Hierarchy but have a direction.*'x'"),
    ],
)
def test_only_hierarchy_classifications_have_a_direction(kind, direction, message):
    classification = pd.DataFrame(
        [(kind, "x", direction)], columns=["predicate_kind", "predicate_subkind", "direction"]
    )

    with pytest.raises(ClassificationDataError, match=message):
        check_classification(classification)


def test_an_empty_relationship_vocabulary_fails_clearly():
    with pytest.raises(RuntimeError, match="relationship vocabulary is empty"):
        check_mapping_against_cdm(GOOD, {})
