"""Registry of oaklib methods the OMOP adapter cannot support, and the error they raise."""

from __future__ import annotations

from dataclasses import dataclass

_READ_ONLY = (
    "The OMOP adapter only reads. Vocabulary data is written by loading Athena releases "
    "with orm-loader, and embeddings by omop-emb."
)
_NO_EDGE_INDEX = (
    "Building the edge index loads every concept relationship into memory. "
    "Use relationships(), with include_entailed for entailed ones, and filters instead."
)

_NO_FILE_EDITING = (
    "Editing, comparing, merging and exporting ontology files do not apply, since "
    "vocabularies enter OMOP through Athena loads."
)
_NO_ASSOCIATIONS = "Associations would be clinical table content, which is a different scope."
_NO_CLUSTERS = (
    "Clustering the mappings loads all of them into memory, about 20 million in an Athena "
    "release. Use get_transitive_mappings_by_curie for one concept."
)

_NO_SPANS = (
    "Only whole-text grounding is supported, so set matches_whole_text. "
    "Spans within a longer text are not annotated."
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
    "MappingProviderInterface.get_mapping_clusters": _NO_CLUSTERS,
    "TextAnnotatorInterface.annotate_text": _NO_SPANS,
}
"""Reason for each unsupported oaklib method, keyed by ``Interface.method``."""


@dataclass(frozen=True)
class UnclaimedInterface:
    """
    An oaklib interface the adapter does not implement.

    Parameters
    ----------
    reason : str
        Why the adapter does not implement it.
    planned : bool
        True when the interface is meant to be implemented later.
    """

    reason: str
    planned: bool = False


NOT_CLAIMED: dict[str, UnclaimedInterface] = {
    "SemanticSimilarityInterface": UnclaimedInterface(
        "Structural similarity from the ancestry, with the cosine similarity from omop-emb.",
        planned=True,
    ),
    "EmbeddingProviderInterface": UnclaimedInterface(
        "Vectors from omop-emb, an optional extra.", planned=True
    ),
    "SubsetterInterface": UnclaimedInterface(
        "Gap filling from the ancestry. Extracting a subset needs an in-memory ontology.",
        planned=True,
    ),
    "SummaryStatisticsInterface": UnclaimedInterface(
        "Counts over concepts and relationships.", planned=True
    ),
    "MetadataInterface": UnclaimedInterface(
        "OMOP has no contributors or creators, and annotation statements are thin.", planned=True
    ),
    "SkosInterface": UnclaimedInterface(
        "It fits OMOP, but no oaklib implementation uses it.", planned=True
    ),
    "ValidatorInterface": UnclaimedInterface("Its OBO lint checks have no OMOP meaning."),
    "OwlInterface": UnclaimedInterface("OMOP has no axioms and no OWL reasoner."),
    "RdfInterface": UnclaimedInterface("OMOP has no triple store."),
    "OboLegacyInterface": UnclaimedInterface("It handles OBO shorthand identifiers."),
    "PatcherInterface": UnclaimedInterface(_NO_FILE_EDITING),
    "DifferInterface": UnclaimedInterface(_NO_FILE_EDITING),
    "MergeInterface": UnclaimedInterface(_NO_FILE_EDITING),
    "DumperInterface": UnclaimedInterface(_NO_FILE_EDITING),
    "OntologyGenerationInterface": UnclaimedInterface(_NO_FILE_EDITING),
    "TaxonConstraintInterface": UnclaimedInterface("OMOP has no taxa."),
    "AssociationProviderInterface": UnclaimedInterface(_NO_ASSOCIATIONS),
    "ClassEnrichmentCalculationInterface": UnclaimedInterface(_NO_ASSOCIATIONS),
    "UsagesInterface": UnclaimedInterface("It reports usages of a term in other ontologies."),
}
"""The oaklib interfaces the adapter does not implement, keyed by class name."""


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
