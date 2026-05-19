import logging
import re
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from linkml_runtime.linkml_model.annotations import Annotation
from oaklib.datamodels.search import (
    SearchConfiguration,
    SearchProperty,
    SearchTermSyntax,
)
from oaklib.datamodels.text_annotator import (
    TextAnnotation,
    TextAnnotationConfiguration,
)
from oaklib.datamodels.vocabulary import (
    HAS_DBXREF,
    HAS_EXACT_SYNONYM,
    LABEL_PREDICATE,
)
from oaklib.interfaces import SearchInterface, TextAnnotatorInterface
from oaklib.interfaces.basic_ontology_interface import (
    ALIAS_MAP,
    METADATA_MAP,
    BasicOntologyInterface,
)
from oaklib.interfaces.text_annotator_interface import nen_annotation
from oaklib.types import CURIE, PRED_CURIE

from omop_graph.graph import (
    KnowledgeGraph, 
    KnowledgeGraphEmbeddingConfiguration
)
from omop_graph.extensions.omop_alchemy import ClassIDEnum
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.nodes import LabelMatchKind
from omop_graph.reasoning.grounding import GroundingConstraints, ground_term
from omop_graph.reasoning.resolvers.resolver_pipeline import ResolverPipeline
from omop_graph.render import bind_default_renderers
from omop_graph.utils.text_utils import cava_tokenizer
from omop_graph.oaklib_interface.omop_resource import OMOPOntologyResource
from omop_graph.oaklib_interface.omop_factory import omop_resource


from sqlalchemy import create_engine
from sqlalchemy.engine import URL

logger = logging.getLogger(__name__)

OMOP_PREFIX = "OMOP"
OMOP_REL_PREFIX = "OMOP_REL"

ANNOTATIONS_SPLIT_CHAR = ","
ANNOTATIONS_PARENT_ID_KEY = "parent_ids"
ANNOTATIONS_VOCABS_KEY = "vocabs"
ANNOTATIONS_DOMAINS_KEY = "domains"

SUPPORTED_PROPERTIES = {
    SearchProperty.LABEL.text,
    SearchProperty.ALIAS.text,
    SearchProperty.IDENTIFIER.text,
    SearchProperty.MAPPED_IDENTIFIER.text,
}


def _normalise_properties(config: SearchConfiguration) -> list[str]:
    """
    Extract property text values from a search configuration.

    Parameters
    ----------
    config : SearchConfiguration
        The search configuration object.

    Returns
    -------
    list[str]
        A list of property strings.
    """
    return [
        p.text if hasattr(p, "text") else str(p) for p in (config.properties or [])  # type: ignore
    ]


class OMOPBaseInterface:
    """
    Base class for OMOP interfaces providing CURIE parsing and Prefix Mapping.

    Parameters
    ----------
    kg : KnowledgeGraph
        The OMOP Knowledge Graph instance.
    """

    def __init__(self, kg: KnowledgeGraph, **kwargs):
        super().__init__(**kwargs)
        self.kg = kg

    def prefix_map(self) -> Dict[str, str]:
        """
        Return the prefix map for OMOP entities.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping prefixes to URI bases.
        """
        return {
            OMOP_PREFIX: "https://athena.ohdsi.org/concept/",
            OMOP_REL_PREFIX: "https://athena.ohdsi.org/relationship/",
        }

    def _concept_curie(self, concept_id: int) -> CURIE:
        """
        Convert a raw OMOP concept ID to a CURIE.

        Parameters
        ----------
        concept_id : int
            The OMOP Concept ID.

        Returns
        -------
        str
            The formatted CURIE (e.g., 'OMOP:12345').

        Raises
        ------
        AssertionError
            If `concept_id` is not an integer.
        """
        assert isinstance(concept_id, int), "Expected concept_id to be an integer"
        return ":".join((OMOP_PREFIX, str(concept_id)))

    def _parse_concept(self, curie: CURIE) -> int:
        """
        Parse a CURIE into an OMOP concept ID.

        If the CURIE uses the `OMOP` prefix, the local part is parsed as an integer.
        Otherwise, it is treated as `vocab:code` and looked up in the Knowledge Graph.

        Parameters
        ----------
        curie : CURIE
            The compact URI string.

        Returns
        -------
        int
            The resolved OMOP concept ID.
        """
        prefix, _, local = curie.partition(":")

        if prefix == OMOP_PREFIX:
            return int(local)

        # treat everything else as vocab:code
        vocab = prefix
        code = local
        return self.kg.concept_id_by_code(vocab, code)

    def _predicate_curie(self, relationship_id: str | int) -> PRED_CURIE:
        """
        Convert a relationship ID to a predicate CURIE.

        Parameters
        ----------
        relationship_id : str | int
            The OMOP relationship ID.

        Returns
        -------
        PRED_CURIE
            The formatted predicate CURIE.
        """
        return ":".join((OMOP_REL_PREFIX, str(relationship_id)))

    def _parse_predicate(self, pred: PRED_CURIE) -> str:
        """
        Parse a predicate CURIE into a raw relationship ID.

        Parameters
        ----------
        pred : PRED_CURIE
            The predicate CURIE.

        Returns
        -------
        str
            The local part of the CURIE (relationship ID).

        Raises
        ------
        ValueError
            If the prefix does not match `OMOP_REL_PREFIX`.
        """
        prefix, _, local = pred.partition(":")
        if prefix != OMOP_REL_PREFIX:
            raise ValueError(f"Unsupported predicate CURIE: {pred}")
        return local

    def precompute_lookups(self) -> None:
        """
        Pre-cache concept views for root nodes in the Knowledge Graph.
        """
        # Warm vocab/domain roots (cheap enough)
        for cid in self.kg.roots():
            self.kg.concept_view(self._concept_curie(cid))


class OMOPTextAnnotatorInterface(OMOPBaseInterface, TextAnnotatorInterface):
    """
    Mixin providing text annotation capabilities via a configurable tokenizer.

    Parameters
    ----------
    kg : KnowledgeGraph
        The underlying OMOP knowledge graph.
    tokenizer : str, optional
        The tokenizer strategy ('simple' or 'cava'). Default is 'simple'.
    """

    def __init__(
        self, kg: KnowledgeGraph, tokenizer: str = "simple", **kwargs
    ):
        super().__init__(kg=kg, **kwargs)
        if tokenizer == "cava":
            self.tokenizer = cava_tokenizer()
        else:
            self.tokenizer = self._simple_tokenizer

    def _simple_tokenizer(self, text: str):
        for m in re.finditer(r"\b[\w\- ]{3,}\b", text):
            yield m.start(), m.end(), m.group()
    
    def annotate_text(
        self,
        text: str,
        query_embedding: Optional[np.ndarray] = None,
        configuration: Optional[TextAnnotationConfiguration] = None,
        annotations: Optional[Dict[str, Annotation]] = None,
    ) -> Iterator[TextAnnotation]:
        """
        Annotate text by grounding terms to the OMOP vocabulary.

        Parameters
        ----------
        text : str
            The input text to annotate.
        query_embedding : np.ndarray
            Pre-computed query embedding for the input text. When None and the KG
            has a writer interface, the embedding is computed on demand.
        configuration : TextAnnotationConfiguration, optional
            Configuration settings for annotation (e.g., token exclusion).
        annotations : Dict[str, Annotation], optional
            LinkML annotations that provide context constraints (parent IDs,
            vocabularies, domains).

        Yields
        -------
        TextAnnotation
            An object representing the grounded text span.

        Raises
        ------
        RuntimeError
            If grounding fails for the text.
        Exception
            If annotations are malformed.
        """
        if configuration is None:
            configuration = TextAnnotationConfiguration()

        if isinstance(text, str) and configuration.token_exclusion_list:
            text = " ".join(
                [
                    term
                    for term in text.split()
                    if term not in configuration.token_exclusion_list
                ]
            )
        elif isinstance(text, tuple) and configuration.token_exclusion_list:
            filtered_text = tuple()
            # text is a tuple of string(s)
            for token in text:
                filtered_text = filtered_text + tuple(
                    term
                    for term in token.split()
                    if term not in configuration.token_exclusion_list
                )
            text = " ".join(filtered_text)

        if annotations is not None:

            def split_annotations(ann):
                if ann is not None and isinstance(ann.value, str):
                    return tuple(val.strip() for val in ann.value.split(ANNOTATIONS_SPLIT_CHAR))
                return None

            try:
                parent_ids = split_annotations(
                    annotations.get(ANNOTATIONS_PARENT_ID_KEY, None)
                )
                vocabs = split_annotations(
                    annotations.get(ANNOTATIONS_VOCABS_KEY, None)
                )
                domains = split_annotations(
                    annotations.get(ANNOTATIONS_DOMAINS_KEY, None)
                )
            except Exception as e:
                logger.error(
                    f"Annotations passed in wrong format. Expected key: val, "
                    f"where val is separated by '{ANNOTATIONS_SPLIT_CHAR}' if using multiple values."
                )
                raise e
        else:
            parent_ids = None
            vocabs = None
            domains = None

        constraints = GroundingConstraints(
            parent_ids=(
                tuple(self._parse_concept(pid) for pid in parent_ids)
                if parent_ids
                else None
            ),
            search_constraint=SearchConstraintConcept(
                domains=domains,
                vocabularies=vocabs,
                require_standard=parent_ids is None,
            ),
            max_depth=6,
            predicate_kinds=frozenset([ClassIDEnum.IDENTITY]),
        )

        resolver_pipeline = ResolverPipeline.with_all_resolvers()
        grounded = ground_term(
            resolver_pipeline=resolver_pipeline,
            kg=self.kg,
            query=text,
            constraints=constraints,
            query_embedding=query_embedding,
        )

        if not grounded:
            raise RuntimeError(f"Failed to ground text: {text}")
            # TODO: Embedding retrieval only as a fallback?

        for ground in grounded:
            yield nen_annotation(
                text=text,
                object_id=self._concept_curie(ground.concept_id),
                object_label=ground.concept_name,
            )


class OMOPSearchInterface(OMOPBaseInterface, SearchInterface):
    """
    Mixin providing search capabilities over an OMOP knowledge graph.
    """

    def __init__(self, *args, kg: KnowledgeGraph, **kwargs):
        super().__init__(*args, kg=kg, **kwargs)

    def basic_search(
        self,
        search_term: str,
        config: Optional[SearchConfiguration] = None,
    ) -> Iterable[CURIE]:
        """
        Perform a basic search over the knowledge graph.

        Parameters
        ----------
        search_term : str
            The term to search for.
        config : SearchConfiguration, optional
            Configuration for search behavior (fuzzy, properties to search).

        Yields
        -------
        CURIE
            The CURIEs of matching concepts.

        Raises
        ------
        ValueError
            If Regex search is requested (not supported).
        """
        if config is None:
            config = SearchConfiguration()

        props = _normalise_properties(config)

        if SearchProperty.ANYTHING.text in props:
            props = list(SUPPORTED_PROPERTIES)

        props = [p for p in props if p in SUPPORTED_PROPERTIES]

        seen: set[int] = set()

        if config.syntax == SearchTermSyntax.REGULAR_EXPRESSION:
            raise ValueError("REGULAR_EXPRESSION search is not supported for OMOP")
        
        if config.is_partial or config.syntax == SearchTermSyntax.STARTS_WITH:
            match_kind = LabelMatchKind.PARTIAL
        else:
            match_kind = LabelMatchKind.EXACT

        if SearchProperty.LABEL.text in props:
            matches = self.kg.concept_lookup(
                search_term,
                match_kind=match_kind,
                synonym=False,
            )
            for lm in matches:
                cid = lm.matched_concept_id
                if cid not in seen:
                    seen.add(cid)
                    yield self._predicate_curie(cid)

        if SearchProperty.ALIAS.text in props:
            matches = self.kg.concept_lookup(
                search_term,
                match_kind=match_kind,
                synonym=True,
            )
            for lm in matches:
                cid = lm.matched_concept_id
                if cid not in seen:
                    seen.add(cid)
                    yield self._predicate_curie(cid)

        if (
            SearchProperty.IDENTIFIER.text in props
            or SearchProperty.MAPPED_IDENTIFIER.text in props
        ):
            if ":" in search_term:
                vocab, code = search_term.split(":", 1)
                try:
                    cid = self.kg.concept_id_by_code(vocab, code)
                    if cid not in seen:
                        seen.add(cid)
                        yield self._predicate_curie(cid)
                except Exception:
                    pass


class OMOPRelationGraphInterface(OMOPBaseInterface, BasicOntologyInterface):
    """
    Mixin providing relation graph capabilities over an OMOP knowledge graph.

    This adapter does not perform OWL reasoning. Entailment is computed via
    graph traversal over hierarchical predicates only.
    """

    def __init__(self, *args, kg: KnowledgeGraph, **kwargs):
        super().__init__(*args, kg=kg, **kwargs)

    def supports_reasoning(self) -> bool:
        return False

    def entity_aliases(self, curie: CURIE) -> Iterable[str]:
        """
        Retrieve aliases (synonyms and codes) for a given entity.

        Parameters
        ----------
        curie : CURIE
            The entity identifier.

        Returns
        -------
        Iterable[str]
            A sorted list of aliases.
        """
        cid = self._parse_concept(curie)
        cv = self.kg.concept_view(cid)
        aliases = set()

        # preferred label
        aliases.add(cv.concept_name)
        # OMOP synonyms
        aliases.update(self.kg.synonyms_for_concept(cid))
        # vocabulary-qualified code alias
        aliases.add(f"{cv.vocabulary_id}:{cv.concept_code}")

        return sorted(aliases)

    def parents(self, curie: CURIE) -> Iterable[CURIE]:
        """
        Retrieve direct parents of the concept.
        """
        concept_id = self._parse_concept(curie)
        for parent_id in self.kg.parents(concept_id):
            yield self._concept_curie(parent_id)

    def children(self, curie: CURIE) -> Iterable[CURIE]:
        """
        Retrieve direct children of the concept.
        """
        concept_id = self._parse_concept(curie)
        for parent_id in self.kg.parents(concept_id):
            yield self._concept_curie(parent_id)

    @property
    def default_language(self) -> Optional[str]:
        return "en"

    def ontologies(self) -> Iterable[CURIE]:
        # yield "OMOP"
        raise NotImplementedError("OMOP does not expose ontology-level metadata")

    def languages(self) -> Iterable[str]:
        # return iter(())
        raise NotImplementedError("OMOP does not support multilingual labels")

    @property
    def multilingual(self) -> bool:
        return False

    def entities(
        self,
        domain: str | None = None,
        standard_only: bool = True,
        filter_obsoletes: bool = True,
    ) -> Iterable[CURIE]:
        """
        Iterate over entities in the graph, optionally filtered.

        Notes
        -----
        We are consuming the entire session object to not have an open connection
        that isn't closed.

        Parameters
        ----------
        domain : str | None
            Filter by OMOP Domain ID.
        standard_only : bool
            If True, return only standard concepts.
        filter_obsoletes : bool
            If True, exclude concepts with an `invalid_reason`.

        Yields
        -------
        CURIE
            Concept identifiers.
        """

        with self.kg.session_factory() as session:
            cids = tuple(self.kg.entities(
                session=session,
                domain=domain,
                standard_only=standard_only,
                filter_obsoletes=filter_obsoletes
            ))
        
        for cid in cids:
            yield self._concept_curie(cid)


    def roots(
        self,
        predicates=None,
        ignore_owl_thing=True,
        filter_obsoletes=True,
        annotated_roots=False,
        id_prefixes=None,
    ) -> Iterable[CURIE]:
        """
        Retrieve root nodes of the graph.
        """
        for cid in self.kg.roots():
            yield self._concept_curie(cid)

    def leafs(
        self,
        predicates: Optional[List[PRED_CURIE]] = None,
        ignore_owl_nothing=True,
        filter_obsoletes=True,
    ) -> Iterable[CURIE]:
        """
        Retrieve leaf nodes of the graph.
        """
        for cid in self.kg.leaves():
            yield self._concept_curie(cid)

    def singletons(
        self, predicates: Optional[List[PRED_CURIE]] = None, filter_obsoletes=True
    ) -> Iterable[CURIE]:
        """
        Retrieve singleton nodes (no parents or children).
        """
        for cid in self.kg.singletons():
            yield self._concept_curie(cid)

    def label(self, curie: CURIE, lang: Optional[str] = None) -> Optional[str]:
        """
        Get the preferred label (concept_name) for a CURIE.
        """
        concept_id = self._parse_concept(curie)
        return self.kg.concept_view(concept_id).concept_name

    def curies_by_label(self, label: str) -> List[CURIE]:
        """
        Retrieve CURIEs that match the exact label.
        """
        # Prefer exact concept_name matches
        cids = self.kg.concept_ids_by_label(label.strip())
        return [self._concept_curie(cid) for cid in cids]

    def relationships(
        self,
        subjects: list[CURIE] | None = None,
        predicates: list[str] | None = None,
        objects: list[CURIE] | None = None,
        invert: bool = False,
    ) -> Iterable[Tuple[CURIE, PRED_CURIE, CURIE]]:
        """
        Query relationships between concepts.

        Parameters
        ----------
        subjects : list[CURIE] | None
            List of subject CURIEs.
        predicates : list[str] | None
            List of predicate (relationship) IDs.
        objects : list[CURIE] | None
            List of object CURIEs.
        invert : bool
            If True, swaps subjects and objects in the query and result.

        Yields
        -------
        Tuple[CURIE, PRED_CURIE, CURIE]
            Triples (subject, predicate, object).
        """
        
        subject_ids = tuple([self._parse_concept(s) for s in subjects]) if subjects is not None else None
        predicate_ids = tuple(predicates) if predicates is not None else None
        object_ids = tuple([self._parse_concept(o) for o in objects]) if objects is not None else None
        
        with self.kg.session_factory() as session:
            relationships = tuple(self.kg.relationships(
                session=session,
                subjects=subject_ids,
                predicates=predicate_ids,
                objects=object_ids,
            ))

        for s,p,o in relationships:
            yield (
                self._concept_curie(s),
                p,
                self._concept_curie(o),
            )

    def hierarchical_parents(
        self, curie: CURIE, isa_only: bool = False
    ) -> List[CURIE]:
        """
        Get hierarchical parents.
        """
        cid = self._parse_concept(curie)
        parents = self.kg.parents(cid)
        return [self._concept_curie(p) for p in parents]

    def simple_mappings_by_curie(self, curie: CURIE):
        raise NotImplementedError(
            "TODO: need to implement mapping interface and have self.sssom_mappings"
        )

    def entity_alias_map(self, curie: CURIE) -> ALIAS_MAP:
        """
        Get a map of alias types to alias values.
        """
        cid = self._parse_concept(curie)
        cv = self.kg.concept_view(cid)

        m = defaultdict(list)

        m[LABEL_PREDICATE].append(cv.concept_name)

        for s in self.kg.synonyms_for_concept(cid):
            m[HAS_EXACT_SYNONYM].append(s)

        m[HAS_DBXREF].append(f"{cv.vocabulary_id}:{cv.concept_code}")

        return m

    def entity_metadata_map(self, curie: CURIE) -> METADATA_MAP:
        """
        Retrieve metadata (OMOP fields) for a concept.
        """
        cid = self._parse_concept(curie)
        cv = self.kg.concept_view(cid)

        m: METADATA_MAP = defaultdict(list)

        # required
        m["id"].append(curie)
        m["label"].append(cv.concept_name)

        # core OMOP metadata
        m["omop:concept_id"].append(f"{cv.concept_id}")
        m["omop:domain"].append(cv.domain_id)
        m["omop:vocabulary"].append(cv.vocabulary_id)
        m["omop:concept_class"].append(cv.concept_class_id)
        m["omop:concept_code"].append(cv.concept_code)

        if cv.standard_concept:
            m["omop:standard_concept"].append("S")

        # validity
        m["omop:valid_start_date"].append(f"{cv.valid_start_date}")
        m["omop:valid_end_date"].append(f"{cv.valid_end_date}")

        if cv.invalid_reason:
            m["omop:invalid_reason"].append(cv.invalid_reason)

        # vocabulary-qualified identifier (important!)
        m["xref"].append(f"{cv.vocabulary_id}:{cv.concept_code}")

        return dict(m)

    def entailed_outgoing_relationships(
        self,
        curie: CURIE,
        predicates: list[PRED_CURIE] | None = None,
    ) -> Iterable[Tuple[PRED_CURIE, CURIE]]:
        """
        Retrieve outgoing relationships, including those implied by the hierarchy.
        """
        raise NotImplementedError(
            "Changes to the CDM currently prevents this function"
        )
        concept_id = self._parse_concept(curie)

        pred_filter = (
            {self._parse_predicate(p) for p in predicates} if predicates else None
        )

        for edge in self.kg.iter_edges(
            concept_id, direction="out", predicate_kinds=None
        ):
            if pred_filter and edge.predicate_id not in pred_filter:
                continue

            pred_curie = self._predicate_curie(edge.predicate_id)

            # hierarchical entailment
            if self.kg.predicate_kind(edge.predicate_id) == ClassIDEnum.HIERARCHY:
                yield pred_curie, self._concept_curie(edge.object_id)

                for parent in self.kg.parents(edge.object_id):
                    yield pred_curie, self._concept_curie(parent)

            else:
                yield pred_curie, self._concept_curie(edge.object_id)

    def entailed_outputgoing_relationships_by_curie(
        self, *args, **kwargs
    ) -> Iterable[Tuple[PRED_CURIE, CURIE]]:
        # return self.entailed_outgoing_relationships(*args, **kwargs)
        raise NotImplementedError(
            "Not Implemented: use entailed_outgoing_relationships instead"
        )

    def entailed_incoming_relationships(
        self,
        curie: CURIE,
        predicates: list[PRED_CURIE] | None = None,
    ) -> Iterable[Tuple[PRED_CURIE, CURIE]]:
        """
        Retrieve incoming relationships.
        """
        concept_id = self._parse_concept(curie)

        pred_filter = (
            {self._parse_predicate(p) for p in predicates} if predicates else None
        )

        with self.kg.session_factory() as session:
            for edge in self.kg.iter_edges(
                session=session,
                concept_ids=concept_id,
                direction="in",
                predicate_ids=frozenset(pred_filter) if pred_filter else None,
            ):
                yield (
                    self._predicate_curie(edge.predicate_id),
                    self._concept_curie(edge.subject_id),
                )

    def entailed_incoming_relationships_by_curie(
        self, *args, **kwargs
    ) -> Iterable[Tuple[PRED_CURIE, CURIE]]:
        # return self.entailed_incoming_relationships(*args, **kwargs)
        raise NotImplementedError(
            "Not Implemented: use entailed_incoming_relationships instead"
        )

    def entailed_relationships_between(
        self,
        subject: CURIE,
        object: CURIE,
    ) -> Iterable[PRED_CURIE]:
        """
        Find relationships connecting a subject and object, including hierarchical ones.
        """
        raise NotImplementedError("Change in OMOP CDM made this function not work anymore")

        subj_id = self._parse_concept(subject)
        obj_id = self._parse_concept(object)

        # direct relationships
        for edge in self.kg.iter_edges(subj_id, direction="out"):
            if edge.object_id == obj_id:
                yield self._predicate_curie(edge.predicate_id)

        # hierarchical entailment
        if obj_id in self.kg.parents(subj_id):
            yield self._predicate_curie("is_a")


class OMOPAlchemyImplementation(
    OMOPRelationGraphInterface,
    OMOPSearchInterface,
    OMOPTextAnnotatorInterface,
):
    """
    A :class:`OntologyInterface` implementation wrapping a SQL Relational Database
    conforming to the OMOP CDM.

    To connect, either use OMOPAlchemyImplementation directly:
    >>> from omop_spires.implementation.omop_implementation import OMOPAlchemyImplementation
    >>> from omop_spires.resource import omop_resource
    >>> resource = omop_resource(url='postgresql+psycopg2://uid:pid@host:5432/dbname')
    >>> adapter = OMOPAlchemyImplementation(resource=resource)

    or
    >>> from omop_spires import get_adapter
    >>> adapter = get_adapter("sqlite://///path/to/your/sqlite/test.db")

    Parameters
    ----------
    engine_string : str | URL | None, optional
        The database connection string.
    resource : OMOPOntologyResource | None, optional
        An existing resource object.
    kg : KnowledgeGraph | None, optional
        An existing Knowledge Graph instance. If None, one is created.
    kg_emb_backend : EmbeddingBackendName, optional
        Optional embedding backend for ``KnowledgeGraph`` construction when
        ``kg`` is not provided.
        Resolution order:
        1. explicit ``kg_emb_backend`` argument
        2. ``OMOP_EMB_BACKEND`` environment variable (inside ``omop_emb``)
        If both are missing, embedding initialization fails only when embedding
        operations are accessed.
    kg_emb_base_storage_dir : str | None, optional
        Optional base directory forwarded to the embedding backend constructor.
        Typical resolution order:
        1. explicit ``kg_emb_base_storage_dir`` argument
        2. ``OMOP_EMB_BASE_STORAGE_DIR`` environment variable
        3. backend default directory
    """

    def __init__(
        self,
        engine_string: str | URL | None = None,
        resource: OMOPOntologyResource | None = None,
        kg: KnowledgeGraph | None = None,
        kg_emb_config: Optional[KnowledgeGraphEmbeddingConfiguration] = None,
        **kwargs,
    ):
        if engine_string is not None:
            self.engine_string = engine_string
            self.resource = resource or omop_resource(url=self.engine_string)
        else:
            load_dotenv()
            self.resource = resource or omop_resource()
            self.engine_string = self.resource.url

        assert self.engine_string is not None, "No database URL provided for OMOPAlchemyImplementation"
        
        engine = create_engine(self.engine_string, future=True, echo=False)

        self._connection = None

        if kg is None:
            kg = KnowledgeGraph(
                emb_config=kg_emb_config,
                cdm_engine=engine
            )
            bind_default_renderers(kg)
        
        super().__init__(kg=kg, **kwargs)


    # TODO: Implement if necessary!
    def _all_relationships(self):
        raise NotImplementedError(
            "OMOP-backed adapter does not support global relationship enumeration"
        )

    def _all_entailed_relationships(self):
        raise NotImplementedError(
            "OMOP-backed adapter does not support global entailed relationship enumeration"
        )

    @property
    def edge_index(self):
        raise NotImplementedError(
            "edge_index is not supported for OMOP-backed adapters"
        )

    @property
    def entailed_edge_index(self):
        raise NotImplementedError(
            "entailed_edge_index is not supported for OMOP-backed adapters"
        )

    def obsoletes(self, include_merged=True) -> Iterable[CURIE]:
        raise NotImplementedError(
            "OMOP does not support OWL-style obsolescence semantics"
        )

    def dangling(self, curies: Optional[Iterable[CURIE]] = None) -> Iterable[CURIE]:
        raise NotImplementedError("OMOP does not support dangling term semantics")

    def ontology_versions(self, ontology: CURIE) -> Iterable[str]:
        raise NotImplementedError("OMOP does not support ontology version semantics")

    def ontology_metadata_map(self, ontology: CURIE) -> Dict[PRED_CURIE, List[str]]:
        raise NotImplementedError("OMOP does not expose ontology-level metadata")

    def subsets(self) -> Iterable[CURIE]:
        raise NotImplementedError("OMOP does not define ontology subsets")

    def subset_members(self, subset: CURIE) -> Iterable[CURIE]:
        raise NotImplementedError("OMOP does not define ontology subsets")

    def terms_categories(self, curies: Iterable[CURIE]):
        raise NotImplementedError("OMOP does not define ontology categories")

    def multilingual_labels(self, curies, allow_none=True, langs=None):
        raise NotImplementedError("OMOP adapter does not support multilingual labels")

    def set_label(self, curie: CURIE, label: str, lang: str | None = None) -> bool:
        raise NotImplementedError("OMOP adapter is read-only")

    def relationships_metadata(self, relationships, **kwargs):
        raise NotImplementedError("OMOP does not support relationship metadata")

    def create_entity(
        self,
        *args,
        **kwargs,
    ) -> CURIE:
        raise NotImplementedError("OMOP adapter is read-only")

    def delete_entity(
        self, curie: CURIE, label: str | None = None, **kwargs
    ) -> CURIE:
        raise NotImplementedError("OMOP adapter is read-only")

    def save(self):
        raise NotImplementedError("OMOP adapter is read-only")

    def clone(self, resource):
        raise NotImplementedError("Cloning is not supported for OMOP adapters")

    def definition(self, curie: CURIE, lang: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError("OMOP does not support term definitions")

    def comments(self, curies, allow_none=True, lang=None):
        raise NotImplementedError("OMOP does not support term comments")

    def owl_types(self, entities):
        raise NotImplementedError(
            "owl_types is not supported by OMOPAlchemyImplementation. "
            "OMOP concepts are treated as owl:Class and relationships as "
            "owl:ObjectProperty implicitly via RelationGraphInterface."
        )