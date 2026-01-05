
from dataclasses import dataclass
from datetime import date
from typing import Optional
from enum import Enum, auto


@dataclass(frozen=True)
class ConceptView:
    concept_id: int
    concept_name: str
    concept_code: str
    vocabulary_id: str
    domain_id: str
    concept_class_id: str
    standard_concept: Optional[str]
    valid_start_date: date
    valid_end_date: date
    invalid_reason: Optional[str]

    def __repr__(self):
        return (
            f"ConceptView("
            f"id={self.concept_id}, "
            f"{self.vocabulary_id}:{self.concept_code}, "
            f"name={self.concept_name!r})"
        )
    
class LabelMatchKind(Enum):
    DIRECT = auto()
    SYNONYM = auto()

@dataclass(frozen=True)
class LabelMatch:
    input_label: str
    matched_label: str
    concept_id: int

    match_kind: LabelMatchKind
    is_standard: bool
    is_active: bool

    def __lt__(self, other: "LabelMatch") -> bool:
        return label_match_rank(self) < label_match_rank(other)

def label_match_rank(m: LabelMatch) -> tuple:
    """
    Lower is better.
    """
    return (
        not m.is_standard,          # prefer standard
        not m.is_active,            # prefer active
        m.match_kind is LabelMatchKind.SYNONYM,  # prefer direct
    )