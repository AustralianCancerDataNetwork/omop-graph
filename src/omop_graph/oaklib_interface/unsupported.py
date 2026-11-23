"""Registry of oaklib methods the OMOP adapter cannot support, and the error they raise."""

from __future__ import annotations

_READ_ONLY = (
    "The OMOP adapter only reads. Vocabulary data is written by loading Athena releases "
    "with orm-loader, and embeddings by omop-emb."
)
_NO_EDGE_INDEX = (
    "Building the edge index loads every concept relationship into memory. "
    "Use relationships() and entailed_relationships() with filters instead."
)

UNSUPPORTED: dict[str, str] = {
    "BasicOntologyInterface.set_label": _READ_ONLY,
    "BasicOntologyInterface.create_entity": _READ_ONLY,
    "BasicOntologyInterface.delete_entity": _READ_ONLY,
    "BasicOntologyInterface.save": _READ_ONLY,
    "BasicOntologyInterface.clone": _READ_ONLY,
    "BasicOntologyInterface.query": "OMOP has no query language that oaklib can pass through.",
    "OboGraphInterface.load_graph": _READ_ONLY,
    "BasicOntologyInterface.edge_index": _NO_EDGE_INDEX,
    "BasicOntologyInterface.entailed_edge_index": _NO_EDGE_INDEX,
}
"""Reason for each unsupported oaklib method, keyed by ``Interface.method``."""


class OMOPUnsupportedError(NotImplementedError):
    """A method with no OMOP equivalent, carrying the reason from :data:`UNSUPPORTED`."""

    def __init__(self, key: str):
        super().__init__(f"{key} is not supported by the OMOP adapter. {UNSUPPORTED[key]}")
        self.key = key


class ReadOnlyMixin:
    """Raises :class:`OMOPUnsupportedError` for every oaklib method that edits or saves."""

    def set_label(self, *args, **kwargs):
        raise OMOPUnsupportedError("BasicOntologyInterface.set_label")

    def create_entity(self, *args, **kwargs):
        raise OMOPUnsupportedError("BasicOntologyInterface.create_entity")

    def delete_entity(self, *args, **kwargs):
        raise OMOPUnsupportedError("BasicOntologyInterface.delete_entity")

    def save(self, *args, **kwargs):
        raise OMOPUnsupportedError("BasicOntologyInterface.save")

    def clone(self, *args, **kwargs):
        raise OMOPUnsupportedError("BasicOntologyInterface.clone")

    def query(self, *args, **kwargs):
        raise OMOPUnsupportedError("BasicOntologyInterface.query")

    def load_graph(self, *args, **kwargs):
        raise OMOPUnsupportedError("OboGraphInterface.load_graph")
