"""Text annotation over the OMOP knowledge graph."""

import logging
from collections.abc import Iterator
from itertools import islice
from typing import ClassVar

import numpy as np
from linkml_runtime.linkml_model.annotations import Annotation
from oaklib.datamodels.text_annotator import (
    TextAnnotation,
    TextAnnotationConfiguration,
)
from oaklib.interfaces import TextAnnotatorInterface
from oaklib.interfaces.text_annotator_interface import nen_annotation
from omop_alchemy.cdm.query import ConceptFilter

from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph import KnowledgeGraph
from omop_graph.oaklib_interface.base import OMOPBaseInterface
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError
from omop_graph.reasoning.grounding import GroundingConstraints, ground_term
from omop_graph.reasoning.resolvers.resolver_pipeline import ResolverPipeline

logger = logging.getLogger(__name__)

ANNOTATIONS_SPLIT_CHAR = ","
ANNOTATIONS_PARENT_ID_KEY = "parent_ids"
ANNOTATIONS_VOCABS_KEY = "vocabs"
ANNOTATIONS_DOMAINS_KEY = "domains"


class OMOPTextAnnotatorInterface(OMOPBaseInterface, TextAnnotatorInterface):
    """
    Mixin providing text annotation capabilities.

    Parameters
    ----------
    kg : KnowledgeGraph
        The underlying OMOP knowledge graph.
    """

    IGNORED_FIELDS: ClassVar[tuple[str, ...]] = (
        "sources",
        "categories",
        "model",
        "include_aliases",
    )
    """The ``TextAnnotationConfiguration`` fields that have no effect."""

    def __init__(self, kg: KnowledgeGraph, **kwargs):
        super().__init__(kg=kg, **kwargs)

    def _parent_id(self, parent: str) -> int:
        """
        The concept id of a ``parent_ids`` entry, given as an id or as a CURIE.

        Raises
        ------
        ValueError
            If the entry names no concept.
        """
        concept_id = self.codec.resolve_reference(parent)
        if concept_id is None:
            raise ValueError(f"parent_ids entry {parent!r} names no concept.")
        return concept_id

    def annotate_text(
        self,
        text: str,
        configuration: TextAnnotationConfiguration | None = None,
        query_embedding: np.ndarray | None = None,
        annotations: dict[str, Annotation] | None = None,
        max_depth: int = 6,
    ) -> Iterator[TextAnnotation]:
        """
        Annotate text by grounding the whole text as one term to the OMOP vocabulary.

        Parameters
        ----------
        text : str
            The input text to annotate.
        configuration : TextAnnotationConfiguration, optional
            Must set ``matches_whole_text``. ``limit`` caps the annotations and
            ``token_exclusion_list`` drops tokens from the text.
        query_embedding : np.ndarray, optional
            Pre-computed query embedding for the input text. When None and the KG
            has a writer interface, the embedding is computed on demand.
        annotations : Dict[str, Annotation], optional
            LinkML annotations that provide context constraints (parent IDs,
            vocabularies, domains).
        max_depth : int
            Maximum grounding hierarchy depth. Default: 6

        Returns
        -------
        Iterator[TextAnnotation]
            One annotation per grounded concept. Nothing is yielded when the text
            does not ground.

        Raises
        ------
        OMOPUnsupportedError
            If ``matches_whole_text`` is not set, since spans are not annotated.
        Exception
            If annotations are malformed.
        """
        if configuration is None:
            configuration = TextAnnotationConfiguration()
        if not configuration.matches_whole_text:
            raise OMOPUnsupportedError("TextAnnotatorInterface.annotate_text")

        ignored = [field for field in self.IGNORED_FIELDS if getattr(configuration, field)]
        if ignored:
            logger.info("The annotation configuration has no effect for %s.", ", ".join(ignored))

        return islice(
            self._ground_whole_text(text, configuration, query_embedding, annotations, max_depth),
            configuration.limit,
        )

    def _ground_whole_text(
        self,
        text: str,
        configuration: TextAnnotationConfiguration,
        query_embedding: np.ndarray | None,
        annotations: dict[str, Annotation] | None,
        max_depth: int,
    ) -> Iterator[TextAnnotation]:
        """Ground the text, with its excluded tokens removed, as one term."""
        if configuration.token_exclusion_list:
            text = " ".join(
                term for term in text.split() if term not in configuration.token_exclusion_list
            )

        if annotations is not None:

            def split_annotations(ann):
                if ann is not None and isinstance(ann.value, str):
                    return tuple(val.strip() for val in ann.value.split(ANNOTATIONS_SPLIT_CHAR))
                return None

            try:
                parent_ids = split_annotations(annotations.get(ANNOTATIONS_PARENT_ID_KEY, None))
                vocabs = split_annotations(annotations.get(ANNOTATIONS_VOCABS_KEY, None))
                domains = split_annotations(annotations.get(ANNOTATIONS_DOMAINS_KEY, None))
            except Exception:
                logger.error(
                    f"Annotations passed in wrong format. Expected key: val, "
                    f"where val is separated by '{ANNOTATIONS_SPLIT_CHAR}' if using multiple values."
                )
                raise
        else:
            parent_ids = None
            vocabs = None
            domains = None

        constraints = GroundingConstraints(
            parent_ids=tuple(self._parent_id(pid) for pid in parent_ids) if parent_ids else None,
            search_constraint=ConceptFilter(
                domains=domains,
                vocabularies=vocabs,
                require_standard=False,
            ),
            predicate_kinds=frozenset([PredicateKind.IDENTITY]),
            max_depth=max_depth,
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
            logger.info("Nothing grounds %r", text)
            return

        for ground in grounded:
            yield nen_annotation(
                text=text,
                object_id=self.codec.concept_curie(ground.concept_id),
                object_label=ground.concept_name,
            )
