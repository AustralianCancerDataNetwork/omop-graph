"""Registry of oaklib methods the OMOP adapter cannot support, and the error they raise."""

from __future__ import annotations

from dataclasses import dataclass

_READ_ONLY = (
    "The OMOP adapter only reads. Vocabulary data is written by loading Athena releases "
    "with the omop-alchemy CLI, and embeddings by omop-emb."
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

_NO_CLOSURE_SCAN = (
    "The closure models have no vector index, and searching them without candidates would "
    "compare against every concept. Pass candidates, or search the omop-emb model."
)

_IC_COMPUTED = (
    "Information content is computed from the Is a hierarchy, so scores cannot be loaded or set."
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
    "SubsetterInterface.extract_subset_ontology": (
        "Extracting a subset ontology needs an in-memory ontology. "
        "Use gap_fill_relationships, or extract_gap_filled_graph for a graph."
    ),
    "SummaryStatisticsInterface.summary_statistic_description": (
        "No data dictionary of the statistics is bundled."
    ),
    "SemanticSimilarityInterface.load_information_content_scores": _IC_COMPUTED,
    "SemanticSimilarityInterface.set_information_content_scores": _IC_COMPUTED,
    "EmbeddingProviderInterface.nearest_entities": _NO_CLOSURE_SCAN,
    "EmbeddingProviderInterface.nearest_entities_to_vector": _NO_CLOSURE_SCAN,
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
    "MetadataInterface": UnclaimedInterface(
        "OMOP has no definitions, contributors or creators, and no annotated statements, "
        "so every method would be empty."
    ),
    "SkosInterface": UnclaimedInterface(
        "It fits OMOP, with Is a as broader and narrower and the mappings as exact and "
        "related, but nothing in oaklib uses it and the methods already exist as "
        "hierarchical_parents, descendants and sssom_mappings."
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
