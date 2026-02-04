from enum import Enum

class ResolverConfidence(Enum):
    EXACT = 0
    EXACT_SYNONYM = 1
    PARTIAL = 2
    EMBEDDING = 3
    EXTERNAL = 4

    def __lt__(self, other: "ResolverConfidence") -> bool:
        return self.value < other.value