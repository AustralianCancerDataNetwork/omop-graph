from __future__ import annotations
from typing import Protocol, Iterable, Tuple

try:
    from cava_nlp import CaVaLang  # type: ignore

    _HAS_CAVA = True
except ImportError:
    CaVaLang = None
    _HAS_CAVA = False


class Tokenizer(Protocol):
    def __call__(self, text: str) -> Iterable[Tuple[int, int, str]]: ...


def cava_tokenizer():
    if not _HAS_CAVA:
        raise RuntimeError(
            "CaVa NLP support not installed. "
            "Install with `pip install omop-spires[nlp]`"
        )
    return CaVaLang()  # type: ignore
