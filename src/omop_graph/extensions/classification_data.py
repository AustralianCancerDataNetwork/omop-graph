"""Validation of the predicate classification files against each other and the CDM."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


class ClassificationDataError(ValueError):
    """The predicate classification data is inconsistent or does not match the CDM."""


def check_classification_files(classification: pd.DataFrame, mapping: pd.DataFrame) -> None:
    """Check the two classification files for consistency, raising on every problem found.

    Parameters
    ----------
    classification : pd.DataFrame
        Rows of ``predicate_classification.csv`` with ``predicate_kind`` and
        ``predicate_subkind`` columns.
    mapping : pd.DataFrame
        Rows of ``predicate_mapping.csv`` with ``relationship_id``,
        ``predicate_kind`` and ``predicate_subkind`` columns.

    Raises
    ------
    ClassificationDataError
        If a kind or subkind contains non-ASCII text, a relationship id is mapped
        more than once, or a mapped kind and subkind is missing from the
        classification file.
    """
    problems: list[str] = []
    for label, frame in (
        ("predicate_classification.csv", classification),
        ("predicate_mapping.csv", mapping),
    ):
        for column in ("predicate_kind", "predicate_subkind"):
            values = frame[column].dropna().astype(str)
            bad = sorted(set(values[~values.map(str.isascii)]))
            if bad:
                problems.append(f"{label} has non-ASCII text in {column}: {bad}")

    mapped_ids = mapping["relationship_id"].dropna()
    duplicated = sorted(set(mapped_ids[mapped_ids.duplicated(keep=False)]))
    if duplicated:
        problems.append(f"predicate_mapping.csv maps relationship ids more than once: {duplicated}")

    known = set(zip(classification["predicate_kind"], classification["predicate_subkind"]))
    pairs = mapping[["predicate_kind", "predicate_subkind"]].dropna()
    unknown = sorted(set(zip(pairs["predicate_kind"], pairs["predicate_subkind"])) - known)
    if unknown:
        problems.append(
            "predicate_mapping.csv uses kind and subkind pairs missing from "
            f"predicate_classification.csv: {unknown}"
        )

    if problems:
        raise ClassificationDataError("\n".join(problems))


def check_mapping_against_cdm(mapping: pd.DataFrame, cdm_relationships: Mapping[str, str]) -> None:
    """Check the mapping against the CDM ``relationship`` table, raising on every problem found.

    Every mapped ``relationship_id`` must be in the CDM, a row's ``r_name``, when present,
    must equal the CDM ``relationship_name``, and every CDM relationship must have a
    classified row.

    Parameters
    ----------
    mapping : pd.DataFrame
        Rows of ``predicate_mapping.csv`` with ``relationship_id``, ``predicate_kind``,
        ``predicate_subkind`` and, optionally, ``r_name`` columns.
    cdm_relationships : Mapping[str, str]
        CDM ``relationship_id`` to ``relationship_name``.

    Raises
    ------
    RuntimeError
        If the CDM has no relationships.
    ClassificationDataError
        If a mapped relationship does not match the CDM or a CDM relationship has no
        classification.
    """
    if not cdm_relationships:
        raise RuntimeError(
            "The CDM relationship vocabulary is empty; load vocabulary data "
            "before running `relationship-classification`."
        )

    id_by_name: dict[str, str | None] = {}
    for relationship_id, name in cdm_relationships.items():
        id_by_name[name] = None if name in id_by_name else relationship_id

    ids = mapping["relationship_id"].tolist()
    names = mapping["r_name"].tolist() if "r_name" in mapping.columns else [None] * len(ids)
    problems: list[str] = []
    for relationship_id, raw_name in zip(ids, names, strict=True):
        name = raw_name if isinstance(raw_name, str) else None
        if relationship_id not in cdm_relationships:
            owner = id_by_name.get(name) if name is not None else None
            hint = f"; that name belongs to relationship_id {owner!r}" if owner else ""
            problems.append(
                f"relationship_id {relationship_id!r} (r_name {name!r}) is not in the CDM{hint}"
            )
        elif name is not None and name != cdm_relationships[relationship_id]:
            problems.append(
                f"relationship_id {relationship_id!r} has r_name {name!r} but the CDM "
                f"relationship_name is {cdm_relationships[relationship_id]!r}"
            )

    classified = mapping.dropna(subset=["predicate_kind", "predicate_subkind"])
    unclassified = sorted(set(cdm_relationships) - set(classified["relationship_id"]))
    if unclassified:
        problems.append(
            f"{len(unclassified)} CDM relationships have no classification, so the knowledge "
            f"graph would skip their edges: {unclassified}"
        )

    if problems:
        raise ClassificationDataError(
            "predicate_mapping.csv does not match the CDM relationship table:\n"
            + "\n".join(problems)
        )
