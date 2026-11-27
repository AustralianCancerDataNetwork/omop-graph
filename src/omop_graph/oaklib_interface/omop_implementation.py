from typing import ClassVar

from oa_configurator import ResolvedCDMDatabase
from oaklib.interfaces import (
    MappingProviderInterface,
    OboGraphInterface,
    RelationGraphInterface,
    SearchInterface,
    TextAnnotatorInterface,
)
from oaklib.interfaces.basic_ontology_interface import (
    BasicOntologyInterface,
)
from oaklib.interfaces.embedding_provider_interface import EmbeddingProviderInterface
from oaklib.interfaces.semsim_interface import SemanticSimilarityInterface
from oaklib.interfaces.subsetter_interface import SubsetterInterface
from oaklib.interfaces.summary_statistics_interface import SummaryStatisticsInterface
from oaklib.resource import OntologyResource
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.db.session import cdm_session_factory, resolve_cdm_database
from omop_graph.graph import KnowledgeGraph, KnowledgeGraphEmbeddingConfiguration
from omop_graph.graph.queries import DEFAULT_CONCEPT_FILTER
from omop_graph.graph.structural_similarity import ICSource
from omop_graph.oaklib_interface.basic_ontology import OMOPBasicOntologyInterface
from omop_graph.oaklib_interface.embeddings import OMOPEmbeddingProviderInterface
from omop_graph.oaklib_interface.mapping_provider import OMOPMappingProviderInterface
from omop_graph.oaklib_interface.obograph import OMOPOboGraphInterface
from omop_graph.oaklib_interface.omop_resource import OMOPOntologyResource
from omop_graph.oaklib_interface.relation_graph import OMOPRelationGraphInterface
from omop_graph.oaklib_interface.search import OMOPSearchInterface
from omop_graph.oaklib_interface.similarity import OMOPSemanticSimilarityInterface
from omop_graph.oaklib_interface.subsetter import OMOPSubsetterInterface
from omop_graph.oaklib_interface.summary_statistics import OMOPSummaryStatisticsInterface
from omop_graph.oaklib_interface.text_annotator import OMOPTextAnnotatorInterface
from omop_graph.oaklib_interface.unsupported import ReadOnlyMixin
from omop_graph.render import bind_default_renderers


class OMOPAlchemyImplementation(
    ReadOnlyMixin,
    OMOPMappingProviderInterface,
    OMOPOboGraphInterface,
    OMOPRelationGraphInterface,
    OMOPSearchInterface,
    OMOPSemanticSimilarityInterface,
    OMOPEmbeddingProviderInterface,
    OMOPSubsetterInterface,
    OMOPSummaryStatisticsInterface,
    OMOPBasicOntologyInterface,
    OMOPTextAnnotatorInterface,
    SearchInterface,
    SemanticSimilarityInterface,
    EmbeddingProviderInterface,
    SubsetterInterface,
    SummaryStatisticsInterface,
    TextAnnotatorInterface,
    MappingProviderInterface,
    OboGraphInterface,
    RelationGraphInterface,
    BasicOntologyInterface,
):
    """
    A :class:`OntologyInterface` implementation wrapping a SQL Relational Database
    conforming to the OMOP CDM.

    Select it through oaklib with an oa-configurator database name, or omit
    the name to use ``OmopGraphConfig.cdm_db``:

    >>> from oaklib import get_adapter
    >>> adapter = get_adapter("omop:cdm_db")

    or construct it directly from an already-resolved database:

    >>> from omop_graph.db.session import resolve_cdm_database
    >>> adapter = OMOPAlchemyImplementation(resolved=resolve_cdm_database())

    Parameters
    ----------
    resource : OntologyResource | None, optional
        The oaklib resource. Its ``slug`` names the ``[databases.*]`` entry to
        resolve; ``None`` resolves ``OmopGraphConfig.cdm_db``. Ignored when
        ``resolved`` or ``kg`` is given.
    resolved : ResolvedCDMDatabase | None, optional
        An already-resolved CDM database to build the engines from with
        ``create_cdm_engines()``, without registering schema claims. Ignored
        when ``kg`` is given.
    kg : KnowledgeGraph | None, optional
        An existing Knowledge Graph, used as-is. The caller owns whatever
        engine it wraps.
    kg_emb_config : KnowledgeGraphEmbeddingConfiguration | None, optional
        Embedding configuration forwarded to the ``KnowledgeGraph`` constructor.
        Ignored when ``kg`` is given.
    concept_filter : ConceptFilter, optional
        The concepts the adapter enumerates. Defaults to active standard and
        classification concepts.
    standardise : bool, optional
        If True, a non-standard concept stands in for the one standard concept it maps to
        when walking the hierarchy and entailing relationships. A concept that maps to
        several standard concepts raises a ``ValueError``. Defaults to False.
    information_content_source : ICSource, optional
        Where the descendants of the information content are counted, ``ICSource.IS_A`` by
        default. ``ICSource.CONCEPT_ANCESTOR`` is faster in bulk.
    """

    # oaklib's class_resolver keys plugins by class name; this adds the "omop:" scheme.
    synonyms: ClassVar[tuple[str, ...]] = ("omop",)

    def __init__(
        self,
        resource: OntologyResource | None = None,
        *,
        resolved: ResolvedCDMDatabase | None = None,
        kg: KnowledgeGraph | None = None,
        kg_emb_config: KnowledgeGraphEmbeddingConfiguration | None = None,
        concept_filter: ConceptFilter = DEFAULT_CONCEPT_FILTER,
        standardise: bool = False,
        information_content_source: ICSource = ICSource.IS_A,
        **kwargs,
    ):
        if resource is None:
            resource = OMOPOntologyResource()
        if kg is None:
            if resolved is None:
                resolved = resolve_cdm_database(resource.slug)
            kg = KnowledgeGraph(
                cdm_session_factory(resolved=resolved),
                emb_config=kg_emb_config,
            )
            bind_default_renderers(kg)

        super().__init__(
            resource=resource,
            kg=kg,
            concept_filter=concept_filter,
            standardise=standardise,
            information_content_source=information_content_source,
            **kwargs,
        )
