"""Validation of the predicate classification mapping against the CDM relationship table."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .omop_alchemy import HierarchyDirection, PredicateKind


class ClassificationDataError(ValueError):
    """The predicate classification mapping does not match the CDM relationship table."""


def check_classification(classification: pd.DataFrame) -> None:
    """Check that exactly the Hierarchy rows of the classification have a direction.

    Parameters
    ----------
    classification : pd.DataFrame
        Rows of ``predicate_classification.csv`` with ``predicate_kind``, ``predicate_subkind``
        and ``direction`` columns.

    Raises
    ------
    ClassificationDataError
        If a Hierarchy row has no ``up`` or ``down`` direction, or another row has one.
    """
    directions = classification["direction"]
    is_hierarchy = classification["predicate_kind"] == PredicateKind.HIERARCHY.value
    valid = [direction.value for direction in HierarchyDirection]
    missing = classification.loc[is_hierarchy & ~directions.isin(valid), "predicate_subkind"]
    unexpected = classification.loc[~is_hierarchy & directions.notna(), "predicate_subkind"]
    problems = []
    if not missing.empty:
        problems.append(f"Hierarchy subkinds without an up or down direction: {missing.tolist()}")
    if not unexpected.empty:
        problems.append(
            f"subkinds that are not Hierarchy but have a direction: {unexpected.tolist()}"
        )
    if problems:
        raise ClassificationDataError(
            "predicate_classification.csv is invalid:\n" + "\n".join(problems)
        )


def check_mapping_against_cdm(
    mapping: pd.DataFrame,
    relationships: Mapping[str, tuple[str, str]],
) -> None:
    """Check the mapping against the CDM ``relationship`` table, raising on every problem found.

    Every mapped ``relationship_id`` must be in the CDM and mapped once, a row's ``r_name``,
    must equal the CDM ``relationship_name``, every CDM relationship must have
    a classified row, and a Hierarchy relationship and its CDM reverse must both be Hierarchy
    with opposite directions.

    Parameters
    ----------
    mapping : pd.DataFrame
        Rows of ``predicate_mapping.csv`` with ``relationship_id``, ``predicate_kind``,
        ``predicate_subkind``, ``direction`` and ``r_name`` columns.
    relationships : Mapping[str, tuple[str, str]]
        CDM ``relationship_id`` to ``(relationship_name, reverse_relationship_id)``.

    Raises
    ------
    RuntimeError
        If the CDM has no relationships.
    ClassificationDataError
        If the mapping does not match the CDM.
    """
    if not relationships:
        raise RuntimeError(
            "The CDM relationship vocabulary is empty; load vocabulary data "
            "before running `relationship-classification`."
        )

    ids = mapping["relationship_id"].tolist()
    problems: list[str] = []
    for relationship_id, name in zip(ids, mapping["r_name"], strict=True):
        if relationship_id not in relationships:
            problems.append(
                f"relationship_id {relationship_id!r} (r_name {name!r}) is not in the CDM"
            )
        elif name != relationships[relationship_id][0]:
            problems.append(
                f"relationship_id {relationship_id!r} has r_name {name!r} but the CDM "
                f"relationship_name is {relationships[relationship_id][0]!r}"
            )

    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        problems.append(f"relationship ids mapped more than once: {duplicated}")

    classified = mapping.dropna(subset=["predicate_kind", "predicate_subkind"])
    unclassified = sorted(set(relationships) - set(classified["relationship_id"]))
    if unclassified:
        problems.append(
            f"{len(unclassified)} CDM relationships have no classification, so the knowledge "
            f"graph would skip their edges: {unclassified}"
        )

    kinds = dict(
        zip(
            classified["relationship_id"],
            zip(classified["predicate_kind"], classified["direction"], strict=True),
            strict=True,
        )
    )
    for relationship_id, (kind, direction) in kinds.items():
        if kind != PredicateKind.HIERARCHY.value or relationship_id not in relationships:
            continue
        reverse_id = relationships[relationship_id][1]
        if relationship_id > reverse_id:
            continue
        reverse_kind, reverse_direction = kinds.get(reverse_id, (None, None))
        if reverse_kind != PredicateKind.HIERARCHY.value:
            problems.append(
                f"{relationship_id!r} is Hierarchy but its reverse {reverse_id!r} is "
                f"{reverse_kind or 'unclassified'}"
            )
        elif direction == reverse_direction:
            problems.append(
                f"{relationship_id!r} and its reverse {reverse_id!r} are both {direction!r}"
            )

    if problems:
        raise ClassificationDataError(
            "predicate_mapping.csv does not match the CDM relationship table:\n"
            + "\n".join(problems)
        )
