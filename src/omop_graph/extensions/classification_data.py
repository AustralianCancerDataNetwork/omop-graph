"""Reading and validation of the predicate classification files against the CDM relationship table."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .omop_alchemy import HierarchyDirection, PredicateKind

CLASSIFICATION_FILE = "predicate_classification.csv"
MAPPING_FILE = "predicate_mapping.csv"
KIND_COLUMNS = ["predicate_kind", "predicate_subkind"]


class ClassificationDataError(ValueError):
    """The predicate classification files are inconsistent or do not match the CDM relationship table."""


@dataclass(frozen=True)
class ClassificationReport:
    """Result of checking the mapping against a CDM that did not raise.

    Attributes
    ----------
    unclassified : list[str]
        CDM relationship ids without a row in the mapping.
    unknown_ids : list[str]
        Mapped relationship ids that are not in the CDM.
    """

    unclassified: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        """One message per kind of finding, with the ids and their count."""
        messages = []
        if self.unclassified:
            messages.append(
                f"{len(self.unclassified)} CDM relationships have no classification, so the "
                f"knowledge graph skips their edges: {self.unclassified}"
            )
        if self.unknown_ids:
            messages.append(
                f"{len(self.unknown_ids)} classified relationship ids are not in the CDM and "
                f"are not loaded: {self.unknown_ids}"
            )
        return messages


def check_classification(classification: pd.DataFrame) -> None:
    """Check that the classification has one valid row per kind and subkind.

    Parameters
    ----------
    classification : pd.DataFrame
        Rows of ``predicate_classification.csv`` with ``predicate_kind``, ``predicate_subkind``,
        ``description``, ``semantics`` and ``inference`` columns.

    Raises
    ------
    ClassificationDataError
        If a kind is not a ``PredicateKind``, or a kind and subkind is defined more than once.
    """
    problems = []
    valid_kinds = {kind.value for kind in PredicateKind}
    unknown = sorted(set(classification["predicate_kind"]) - valid_kinds)
    if unknown:
        problems.append(f"unknown classes: {unknown}")
    duplicated = classification[classification.duplicated(KIND_COLUMNS, keep=False)]
    for (kind, subkind), rows in duplicated.groupby(KIND_COLUMNS):
        differing = [
            column
            for column in ("description", "semantics", "inference")
            if rows[column].nunique(dropna=False) > 1
        ]
        detail = f", differing in {differing}" if differing else ""
        problems.append(f"{kind!r}/{subkind!r} is defined {len(rows)} times{detail}")
    if problems:
        raise ClassificationDataError(f"{CLASSIFICATION_FILE} is invalid:\n" + "\n".join(problems))


def check_mapping(mapping: pd.DataFrame, classification: pd.DataFrame) -> None:
    """Check the mapping against the classification, which needs no CDM.

    Parameters
    ----------
    mapping : pd.DataFrame
        Rows of ``predicate_mapping.csv`` with ``relationship_id``, ``predicate_kind``,
        ``predicate_subkind`` and ``direction`` columns.
    classification : pd.DataFrame
        Rows of ``predicate_classification.csv``.

    Raises
    ------
    ClassificationDataError
        If a row has a class and subclass that the classification does not define, or if
        exactly the Hierarchy rows do not have an ``up`` or ``down`` direction.
    """
    problems = []
    defined = set(zip(classification["predicate_kind"], classification["predicate_subkind"]))
    pairs = zip(mapping["predicate_kind"], mapping["predicate_subkind"])
    undefined = mapping.loc[[pair not in defined for pair in pairs], "relationship_id"]
    if not undefined.empty:
        problems.append(
            f"{len(undefined)} rows have a class and subclass without a classification: "
            f"{undefined.tolist()}"
        )
    directions = mapping["direction"]
    is_hierarchy = mapping["predicate_kind"] == PredicateKind.HIERARCHY.value
    valid = [direction.value for direction in HierarchyDirection]
    missing = mapping.loc[is_hierarchy & ~directions.isin(valid), "relationship_id"]
    unexpected = mapping.loc[~is_hierarchy & directions.notna(), "relationship_id"]
    if not missing.empty:
        problems.append(
            f"Hierarchy relationships without an up or down direction: {missing.tolist()}"
        )
    if not unexpected.empty:
        problems.append(
            f"relationships that are not Hierarchy but have a direction: {unexpected.tolist()}"
        )
    if problems:
        raise ClassificationDataError(f"{MAPPING_FILE} is invalid:\n" + "\n".join(problems))


def read_classification_files(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and check ``predicate_classification.csv`` and ``predicate_mapping.csv``.

    Parameters
    ----------
    directory : Path
        Directory holding both files. A missing ``direction`` column in the mapping means no
        relationship has a direction.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The classification with ``predicate_kind``, ``predicate_subkind``, ``description``,
        ``semantics`` and ``inference`` columns, and the mapping with ``relationship_id``,
        ``r_name``, ``predicate_kind``, ``predicate_subkind`` and ``direction`` columns.

    Raises
    ------
    NotADirectoryError
        If ``directory`` is not a directory.
    FileNotFoundError
        If one of the two files is missing.
    ClassificationDataError
        If the files are inconsistent. Nothing is read from a database, so a caller can run
        this before it changes any table.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a valid directory.")
    for filename in (CLASSIFICATION_FILE, MAPPING_FILE):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"`{filename}` not found in {directory}")
    renames = {"class": "predicate_kind", "subclass": "predicate_subkind"}
    classification = pd.read_csv(directory / CLASSIFICATION_FILE, encoding="utf-8-sig").rename(
        columns=renames
    )[[*KIND_COLUMNS, "description", "semantics", "inference"]]
    mapping = pd.read_csv(directory / MAPPING_FILE, encoding="utf-8-sig").rename(
        columns={**renames, "r_id": "relationship_id"}
    )
    mapping = mapping.assign(direction=mapping.get("direction"))[
        ["relationship_id", "r_name", *KIND_COLUMNS, "direction"]
    ]
    check_classification(classification)
    check_mapping(mapping, classification)
    return classification, mapping


def check_mapping_against_cdm(
    mapping: pd.DataFrame,
    relationships: Mapping[str, tuple[str, str]],
    *,
    strict: bool = False,
) -> ClassificationReport:
    """Check the mapping against the CDM ``relationship`` table.

    Errors are always raised: an ``r_name`` that differs from the CDM ``relationship_name``,
    a relationship mapped more than once, and a Hierarchy relationship whose CDM reverse is
    not Hierarchy with the opposite direction. A CDM relationship without a row and a mapped
    id that is not in the CDM are findings of the report, and raise under ``strict``.

    Parameters
    ----------
    mapping : pd.DataFrame
        Output of :func:`read_classification_files`.
    relationships : Mapping[str, tuple[str, str]]
        CDM ``relationship_id`` to ``(relationship_name, reverse_relationship_id)``.
    strict : bool, default False
        Raise on the findings of the report as well.

    Returns
    -------
    ClassificationReport
        The unclassified CDM relationships and the mapped ids that are not in the CDM.

    Raises
    ------
    RuntimeError
        If the CDM has no relationships, or none of the mapped ids is in it.
    ClassificationDataError
        If the mapping does not match the CDM.
    """
    if not relationships:
        raise RuntimeError(
            "The CDM relationship vocabulary is empty; load vocabulary data "
            "before running `relationship-classification`."
        )
    ids = mapping["relationship_id"].tolist()
    if not set(ids) & set(relationships):
        raise RuntimeError(
            "None of the relationship mappings match IDs in the CDM vocabulary; "
            "no relationship classifications would be loaded."
        )

    problems: list[str] = []
    for relationship_id, name in zip(ids, mapping["r_name"], strict=True):
        if relationship_id in relationships and name != relationships[relationship_id][0]:
            problems.append(
                f"relationship_id {relationship_id!r} has r_name {name!r} but the CDM "
                f"relationship_name is {relationships[relationship_id][0]!r}"
            )
    duplicated = sorted(
        mapping.loc[mapping["relationship_id"].duplicated(), "relationship_id"].unique()
    )
    if duplicated:
        problems.append(f"relationship ids mapped more than once: {duplicated}")

    kinds = dict(
        zip(
            mapping["relationship_id"],
            zip(mapping["predicate_kind"], mapping["direction"], strict=True),
            strict=True,
        )
    )
    checked: set[frozenset[str]] = set()
    for relationship_id in sorted(kinds):
        kind, direction = kinds[relationship_id]
        if kind != PredicateKind.HIERARCHY.value or relationship_id not in relationships:
            continue
        reverse_id = relationships[relationship_id][1]
        pair = frozenset({relationship_id, reverse_id})
        if pair in checked:
            continue
        checked.add(pair)
        if reverse_id == relationship_id:
            problems.append(f"{relationship_id!r} is Hierarchy but is its own reverse")
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

    report = ClassificationReport(
        unclassified=sorted(set(relationships) - set(ids)),
        unknown_ids=sorted(set(ids) - set(relationships)),
    )
    if strict:
        problems.extend(report.warnings)
    if problems:
        raise ClassificationDataError(
            f"{MAPPING_FILE} does not match the CDM relationship table:\n" + "\n".join(problems)
        )
    return report
