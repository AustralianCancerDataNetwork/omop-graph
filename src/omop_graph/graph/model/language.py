"""The languages of concept synonyms."""

from __future__ import annotations

from enum import Enum


class OMOPLanguage(Enum):
    """
    The language concepts used by ``concept_synonym``, with their ISO 639-1 codes.

    The concepts are SNOMED language concepts. A language concept outside this list has
    no code and its synonyms are not exposed as translations.
    """

    ENGLISH = (4180186, "en")
    SPANISH = (4182511, "es")
    KOREAN = (4175771, "ko")
    CHINESE = (4182948, "zh")
    FRENCH = (4180190, "fr")
    GERMAN = (4182504, "de")
    JAPANESE = (4181524, "ja")
    PORTUGUESE = (4181536, "pt")
    DUTCH = (4182503, "nl")
    HEBREW = (4180047, "he")
    ITALIAN = (4182507, "it")
    SWEDISH = (4175777, "sv")
    DANISH = (4180183, "da")
    ARABIC = (4181374, "ar")

    def __init__(self, concept_id: int, code: str):
        self.concept_id = concept_id
        self.code = code

    @classmethod
    def from_concept_id(cls, concept_id: int) -> OMOPLanguage | None:
        """Return the language with this concept id, or None when it has no code."""
        return next((language for language in cls if language.concept_id == concept_id), None)

    @classmethod
    def from_code(cls, code: str) -> OMOPLanguage | None:
        """Return the language with this ISO 639-1 code, or None when it is unknown."""
        return next((language for language in cls if language.code == code), None)
