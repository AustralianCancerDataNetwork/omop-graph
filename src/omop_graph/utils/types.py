from enum import Enum, auto


# TODO: Get this out of here by preventing circular imports and have it in the resolver base class.
class ResolverConfidence(Enum):
    EXACT = auto()
    EXACT_SYNONYM = auto()
    FULLTEXT = auto()
    FULLTEXT_SYNONYM = auto()
    PARTIAL = auto()
    PARTIAL_SYNONYM = auto()

    def __lt__(self, other: "ResolverConfidence") -> bool:
        return self.value < other.value