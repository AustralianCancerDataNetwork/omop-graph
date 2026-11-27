"""Registry of oaklib methods the OMOP adapter cannot support, and the error they raise."""

from __future__ import annotations

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

_NO_UNKNOWN_MODEL = (
    "The adapter provides the closure models and the stored model of the graph, which "
    "embedding_models() lists."
)

_NO_CLOSURE_TEXT = (
    "The closure models embed a concept from its ancestors, so they cannot embed a text, "
    "which nearest_entities_to_text needs. Use the omop-emb model, or no model when the "
    "graph has an embedding store."
)

_NO_ALL_PATHS = (
    "Listing every path between two concepts is unbounded in the Is a hierarchy. "
    "Ask for the shortest paths, which is the default."
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
    "OboGraphInterface.paths(shortest=False)": _NO_ALL_PATHS,
    "BasicOntologyInterface.edge_index": _NO_EDGE_INDEX,
    "BasicOntologyInterface.entailed_edge_index": _NO_EDGE_INDEX,
    "MappingProviderInterface.get_mapping_clusters": _NO_CLUSTERS,
    "TextAnnotatorInterface.annotate_text(matches_whole_text=False)": _NO_SPANS,
    "SubsetterInterface.extract_subset_ontology": (
        "Extracting a subset ontology needs an in-memory ontology. "
        "Use gap_fill_relationships, or extract_gap_filled_graph for a graph."
    ),
    "SummaryStatisticsInterface.summary_statistic_description": (
        "No data dictionary of the statistics is bundled."
    ),
    "SemanticSimilarityInterface.load_information_content_scores": _IC_COMPUTED,
    "SemanticSimilarityInterface.set_information_content_scores": _IC_COMPUTED,
    "EmbeddingProviderInterface.nearest_entities(model=closure, candidates=None)": _NO_CLOSURE_SCAN,
    "EmbeddingProviderInterface.nearest_entities_to_vector(model=closure, candidates=None)": (
        _NO_CLOSURE_SCAN
    ),
    "EmbeddingProviderInterface.text_embedding(model=closure)": _NO_CLOSURE_TEXT,
    "EmbeddingProviderInterface._fetch_embeddings(model=unknown)": _NO_UNKNOWN_MODEL,
}
"""
Reason for each unsupported oaklib method, keyed by ``Interface.method``.

A method that is unsupported only for some arguments has the condition after its name, as in
``Interface.method(argument=value)``.
"""


NOT_CLAIMED: dict[str, str] = {
    "MetadataInterface": "OMOP has no definitions, contributors or creators, and no annotated statements, "
    "so every method would be empty.",
    "SkosInterface": "It fits OMOP, with Is a as broader and narrower and the mappings as exact and "
    "related, but nothing in oaklib uses it and the methods already exist as "
    "hierarchical_parents, descendants and sssom_mappings.",
    "ValidatorInterface": "Its OBO lint checks have no OMOP meaning.",
    "OwlInterface": "OMOP has no axioms and no OWL reasoner.",
    "RdfInterface": "OMOP has no triple store.",
    "OboLegacyInterface": "It handles OBO shorthand identifiers.",
    "PatcherInterface": _NO_FILE_EDITING,
    "DifferInterface": _NO_FILE_EDITING,
    "MergeInterface": _NO_FILE_EDITING,
    "DumperInterface": _NO_FILE_EDITING,
    "OntologyGenerationInterface": _NO_FILE_EDITING,
    "TaxonConstraintInterface": "OMOP has no taxa.",
    "AssociationProviderInterface": _NO_ASSOCIATIONS,
    "ClassEnrichmentCalculationInterface": _NO_ASSOCIATIONS,
    "UsagesInterface": "It reports usages of a term in other ontologies.",
}
"""The oaklib interfaces the adapter does not implement, keyed by class name."""


class OMOPUnsupportedError(NotImplementedError):
    """A method with no OMOP equivalent, carrying the reason from :data:`UNSUPPORTED`."""

    def __init__(self, key: str):
        super().__init__(f"{key} is not supported by the OMOP adapter. {UNSUPPORTED[key]}")
        self.key = key


class OMOPUnsupportedPropertyError(OMOPUnsupportedError, AttributeError):
    """
    A property with no OMOP equivalent.

    It is also an ``AttributeError``, so ``hasattr`` and ``inspect.getmembers`` treat the
    property as absent and do not raise.
    """


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
