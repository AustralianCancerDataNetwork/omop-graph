"""
Node definitions for the OMOP Knowledge Graph.

This module defines lightweight data structures (Views) representing entities
in the OMOP graph, such as Concepts and search matches.

These classes are primarily used for:
1.  Holding data returned by database queries.
2.  Rendering rich representations in Jupyter notebooks via `_repr_html_`.
3.  Ranking and sorting search results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum, StrEnum
from html import escape

from omop_alchemy.cdm.model import normalised_flag
from sqlalchemy import Row


@dataclass(frozen=True)
class VocabularyView:
    """
    A row of the `vocabulary` table.

    Parameters
    ----------
    vocabulary_id : str
        The vocabulary identifier (e.g., 'SNOMED').
    vocabulary_name : str
        The full name of the vocabulary.
    vocabulary_reference : str, optional
        Where the vocabulary comes from.
    vocabulary_version : str, optional
        The loaded version of the vocabulary.
    vocabulary_concept_id : int
        The concept that stands for the vocabulary.
    """

    vocabulary_id: str
    vocabulary_name: str
    vocabulary_reference: str | None
    vocabulary_version: str | None
    vocabulary_concept_id: int


@dataclass(frozen=True)
class SynonymView:
    """
    A row of the `concept_synonym` table.

    Parameters
    ----------
    concept_id : int
        The concept the synonym belongs to.
    name : str
        The synonym text.
    language_concept_id : int
        The concept for the language of the synonym.
    """

    concept_id: int
    name: str
    language_concept_id: int


class MappingCardinality(StrEnum):
    """
    How many subjects and objects a mapping connects, with the values of the SSSOM
    ``mapping_cardinality``.

    The first number is the subjects of the object, the second the objects of the subject.
    Most subjects mapped to one object is ``n:1``, and one subject mapped to most objects
    ``1:n``.
    """

    ONE_TO_ONE = "1:1"
    ONE_TO_MANY = "1:n"
    MANY_TO_ONE = "n:1"
    MANY_TO_MANY = "n:n"
    ONE_TO_NONE = "1:0"
    NONE_TO_ONE = "0:1"
    NONE_TO_NONE = "0:0"

    @classmethod
    def of(cls, subjects_of_object: int, objects_of_subject: int) -> MappingCardinality:
        """The cardinality of a mapping whose ends are shared with that many others."""

        def side(count: int) -> str:
            return "n" if count > 1 else str(count)

        return cls(f"{side(subjects_of_object)}:{side(objects_of_subject)}")


@dataclass(frozen=True)
class MappingView:
    """
    A concept relationship together with the vocabularies of its two concepts.

    Parameters
    ----------
    subject_id : int
        The concept the relationship starts from.
    predicate_id : str
        The relationship id.
    object_id : int
        The concept the relationship points to.
    valid_start_date : date
        When the relationship became valid.
    subject_source : str
        The vocabulary of the subject concept.
    object_source : str
        The vocabulary of the object concept.
    cardinality : MappingCardinality
        Counted from the valid relationships of the same id in the connected database, so
        on a subset of the CDM it is relative to the rows that subset holds.
    """

    subject_id: int
    predicate_id: str
    object_id: int
    valid_start_date: date
    subject_source: str
    object_source: str
    cardinality: MappingCardinality


class Standardness(Enum):
    """
    The standardness flag of a concept, with its value in the ``standard_concept`` column.

    A standard concept is a valid target of a mapping. A classification concept is a
    waypoint in the hierarchy and is not a valid mapping target. Any other concept carries
    no flag.
    """

    STANDARD = "S"
    CLASSIFICATION = "C"
    NONE = None


@dataclass(frozen=True)
class ConceptView:
    """
    A lightweight, immutable view of an OMOP Concept.

    This class represents a row from the `concept` table, optimized for
    read-only access and visualization.

    Parameters
    ----------
    concept_id : int
        The unique OMOP identifier.
    concept_name : str
        The human-readable name of the concept.
    concept_code : str
        The source code in the original vocabulary.
    vocabulary_id : str
        The vocabulary identifier (e.g., 'SNOMED', 'RxNorm').
    domain_id : str
        The domain identifier (e.g., 'Condition', 'Drug').
    concept_class_id : str
        The class of the concept (e.g., 'Clinical Finding').
    standard_concept : bool
        True only for a Standard Concept (raw flag ``'S'``). Valid as a mapping
        target.
    classification_concept : bool
        True only for a Classification Concept (raw flag ``'C'``). Valid for
        hierarchy navigation but not as a mapping target. Kept distinct from
        ``standard_concept`` so consumers can tell a classification node from a
        concept carrying no standardness flag at all.
    valid_start_date : date
        The start date of validity.
    valid_end_date : date
        The end date of validity.
    invalid_reason : str, optional
        The reason for invalidation (e.g., 'D', 'U'), or None if valid.
    """

    concept_id: int
    concept_name: str
    concept_code: str
    vocabulary_id: str
    domain_id: str
    concept_class_id: str
    standard_concept: bool
    valid_start_date: date
    valid_end_date: date
    invalid_reason: str | None
    # Defaulted so rows projected before the classification label was added
    # still construct; the query layer always supplies it.
    classification_concept: bool = False

    @property
    def standardness(self) -> Standardness:
        """The standardness flag of the concept, from ``standard_concept`` and ``classification_concept``."""
        if self.standard_concept:
            return Standardness.STANDARD
        if self.classification_concept:
            return Standardness.CLASSIFICATION
        return Standardness.NONE

    @property
    def is_active(self) -> bool:
        """Whether the concept is active under OMOP Alchemy's flag semantics.

        Delegates to ``normalised_flag`` so a blank or whitespace-only
        ``invalid_reason``, which some real-world loads carry instead of NULL,
        is treated as unset, exactly as the SQL-side predicate does.
        """
        return normalised_flag(self.invalid_reason) is None

    def __repr__(self) -> str:
        return (
            f"ConceptView("
            f"id={self.concept_id}, "
            f"{self.vocabulary_id}:{self.concept_code}, "
            f"name={self.concept_name!r})"
        )

    def _repr_html_(self) -> str:
        """
        Render a rich HTML representation for Jupyter notebooks.
        """
        std_badge = ""
        if self.standard_concept:
            std_badge = (
                "<span style='background:#2b7; color:white; padding:2px 6px; "
                "border-radius:4px; font-size:0.75em; margin-left:6px;'>standard</span>"
            )

        inactive_badge = ""
        if self.invalid_reason:
            inactive_badge = (
                "<span style='background:#c33; color:white; padding:2px 6px; "
                "border-radius:4px; font-size:0.75em; margin-left:6px;'>inactive</span>"
            )

        return f"""
        <div style="border:1px solid #ddd; border-radius:8px; padding:8px; max-width:520px;">
          <div style="font-weight:600; font-size:1.05em;">
            {escape(self.concept_name)}
            {std_badge}
            {inactive_badge}
          </div>

          <div style="font-family:monospace; color:#555; margin-top:2px;">
            {escape(self.vocabulary_id)}:{escape(self.concept_code)}
          </div>

          <table style="margin-top:6px; font-size:0.85em;">
            <tr><th align="left">concept_id</th><td>{self.concept_id}</td></tr>
            <tr><th align="left">domain</th><td>{escape(self.domain_id)}</td></tr>
            <tr><th align="left">class</th><td>{escape(self.concept_class_id)}</td></tr>
            <tr><th align="left">valid</th>
                <td>{self.valid_start_date} → {self.valid_end_date}</td></tr>
          </table>
        </div>
        """

    @classmethod
    def from_row(cls, row: Row) -> ConceptView:
        """
        Create a ConceptView from a SQLAlchemy Row.

        The query layer projects OMOP Alchemy's atomic standardness predicates
        into the ``standard_concept`` and ``classification_concept`` fields as
        booleans.

        Parameters
        ----------
        row : Row
            A row object returned by a SQLAlchemy query selecting from the Concept table.

        Returns
        -------
        ConceptView
            The instantiated view.
        """
        data = dict(row._mapping)
        data["standard_concept"] = bool(data["standard_concept"])
        data["classification_concept"] = bool(data.get("classification_concept", False))
        return cls(**data)


@dataclass(frozen=True)
class AncestorMatch:
    """
    Represents a successful hierarchy check between two concepts.

    Parameters
    ----------
    ancestor_concept_id : int
        The concept ID of the ancestor.
    descendant_concept_id : int
        The concept ID of the descendant.
    min_levels_of_separation : int
        The shortest path distance between them (0 = self).
    """

    ancestor_concept_id: int
    descendant_concept_id: int
    min_levels_of_separation: int
