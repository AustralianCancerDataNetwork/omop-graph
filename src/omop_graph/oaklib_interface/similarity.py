"""The semantic similarity interface over the OMOP knowledge graph."""

import logging
from collections.abc import Iterable, Iterator
from typing import ClassVar

from oaklib.datamodels.similarity import (
    InformationContentCorpusEnum,
    InformationContentMethod,
    InformationContentScaleEnum,
    TermPairwiseSimilarity,
)
from oaklib.datamodels.vocabulary import IS_A, OWL_THING
from oaklib.interfaces.semsim_interface import SemanticSimilarityInterface
from oaklib.types import CURIE, PRED_CURIE

from omop_graph.graph.structural_similarity import ICSource
from omop_graph.oaklib_interface.basic_ontology import OMOPBasicOntologyInterface
from omop_graph.oaklib_interface.unsupported import OMOPUnsupportedError

logger = logging.getLogger(__name__)


class OMOPSemanticSimilarityInterface(OMOPBasicOntologyInterface, SemanticSimilarityInterface):
    """
    Mixin serving the structural similarity of concepts from the strict ``Is a`` hierarchy.

    The ancestry, the counts and the scope are those of the knowledge graph, see
    :meth:`omop_graph.graph.kg.KnowledgeGraph.pairwise_similarity`. Only ``rdfs:subClassOf``
    can be followed, so a *predicates* argument must be empty or name it. Information
    content is always computed from the hierarchy, so scores cannot be loaded.
    The ``subject_ancestors`` and ``object_ancestors`` arguments of ``pairwise_similarity``
    have no effect.
    """

    information_content_source: ClassVar[ICSource] = ICSource.IS_A
    """Where the descendants are counted, ``CONCEPT_ANCESTOR`` is faster in bulk."""

    @staticmethod
    def _check_predicates(predicates: Iterable[PRED_CURIE] | None) -> None:
        """Raise unless the predicates are none, or only ``rdfs:subClassOf``."""
        if predicates and set(predicates) != {IS_A}:
            raise ValueError(f"Similarity follows {IS_A} only, got {list(predicates)}.")

    def most_recent_common_ancestors(
        self,
        subject: CURIE,
        object: CURIE,
        predicates: list[PRED_CURIE] | None = None,
        include_owl_thing: bool = True,
    ) -> Iterable[CURIE]:
        """
        Yield the common ancestors that have no descendant among the common ancestors.

        When there are none, ``owl:Thing`` is yielded if *include_owl_thing* is set.
        """
        self._check_predicates(predicates)
        pair = self._resolve_pair(subject, object)
        found = (
            self.kg.most_recent_common_ancestors(*pair, concept_filter=self._population)
            if pair
            else frozenset()
        )
        for concept_id in sorted(found):
            yield self.codec.concept_curie(concept_id)
        if not found and include_owl_thing:
            yield OWL_THING

    def common_ancestors(
        self,
        subject: CURIE,
        object: CURIE,
        predicates: list[PRED_CURIE] | None = None,
        subject_ancestors: list[CURIE] | None = None,
        object_ancestors: list[CURIE] | None = None,
        include_owl_thing: bool = True,
    ) -> Iterable[CURIE]:
        """Yield the concepts that are, or are above, both concepts, and ``owl:Thing`` if asked."""
        self._check_predicates(predicates)
        pair = self._resolve_pair(subject, object)
        found = (
            self.kg.common_ancestors(*pair, concept_filter=self._population)
            if pair
            else frozenset()
        )
        for concept_id in sorted(found):
            yield self.codec.concept_curie(concept_id)
        if include_owl_thing:
            yield OWL_THING

    def common_descendants(
        self,
        subject: CURIE,
        object: CURIE,
        predicates: list[PRED_CURIE] | None = None,
        include_owl_nothing: bool = False,
    ) -> Iterable[CURIE]:
        """Yield the concepts that are, or are below, both concepts."""
        self._check_predicates(predicates)
        if pair := self._resolve_pair(subject, object):
            for concept_id in sorted(
                self.kg.common_descendants(*pair, concept_filter=self._population)
            ):
                yield self.codec.concept_curie(concept_id)

    def load_information_content_scores(self, source: str) -> None:
        """Not supported, the information content is computed from the hierarchy."""
        raise OMOPUnsupportedError("SemanticSimilarityInterface.load_information_content_scores")

    def set_information_content_scores(self, scores: Iterable[tuple[CURIE, float]]) -> None:
        """Not supported, the information content is computed from the hierarchy."""
        raise OMOPUnsupportedError("SemanticSimilarityInterface.set_information_content_scores")

    def information_content_method(
        self,
        object_closure_predicates: list[PRED_CURIE] | None = None,
        use_associations: bool | None = None,
    ) -> InformationContentMethod:
        """Describe the information content as ``-log2`` of the share of concepts below a term."""
        return InformationContentMethod(
            scale=InformationContentScaleEnum.log2_bits.text,
            corpus=InformationContentCorpusEnum.ontology.text,
            closure_predicates=[IS_A],
            source="omop-graph",
        )

    def information_content_scores(
        self,
        curies: Iterable[CURIE] | None = None,
        predicates: list[PRED_CURIE] | None = None,
        object_closure_predicates: list[PRED_CURIE] | None = None,
        use_associations: bool | None = None,
        term_to_entities_map: dict[CURIE, list[CURIE]] | None = None,
        **kwargs,
    ) -> Iterator[tuple[CURIE, float]]:
        """
        Yield each concept with its information content, ``-log2(descendants / concepts)``.

        The descendants include the concept and the concepts are those in scope. Without
        *curies* every concept in scope is scored, which counts the descendants of each.
        Associations are not supported.
        """
        self._check_predicates(object_closure_predicates)
        if use_associations:
            raise ValueError("OMOP has no associations to compute information content from.")
        batches = (
            self.kg.batches(self.entities())
            if curies is None
            else self.kg.batches(dict.fromkeys(curies))
        )
        for batch in batches:
            ids = self._resolve_concepts(batch)
            scores = self.kg.information_content(
                ids.values(), self.information_content_source, self._population
            )
            for curie, concept_id in ids.items():
                yield curie, scores[concept_id]

    def pairwise_similarity(
        self,
        subject: CURIE,
        object: CURIE,
        predicates: list[PRED_CURIE] | None = None,
        subject_ancestors: list[CURIE] | None = None,
        object_ancestors: list[CURIE] | None = None,
        min_jaccard_similarity: float | None = None,
        min_ancestor_information_content: float | None = None,
    ) -> TermPairwiseSimilarity | None:
        """
        Score the similarity of two concepts, or None when it is below a minimum.

        Parameters
        ----------
        subject : CURIE
            The first concept.
        object : CURIE
            The second concept.
        predicates : list[PRED_CURIE] | None
            Empty, or ``rdfs:subClassOf``.
        subject_ancestors : list[CURIE] | None
            Ignored.
        object_ancestors : list[CURIE] | None
            Ignored.
        min_jaccard_similarity : float | None
            The least Jaccard similarity of a pair that is scored.
        min_ancestor_information_content : float | None
            The least information content of the best common ancestor of a scored pair.

        Returns
        -------
        TermPairwiseSimilarity | None
            The similarity, with no ancestor when the concepts share none. None when a
            CURIE names no concept, or a minimum is not met.
        """
        self._check_predicates(predicates)
        pair = self._resolve_pair(subject, object)
        if pair is None:
            return None
        similarity = self.kg.pairwise_similarity(
            *pair, self.information_content_source, self._population
        )
        if min_jaccard_similarity is not None and similarity.jaccard < min_jaccard_similarity:
            return None
        if (
            min_ancestor_information_content is not None
            and similarity.ancestor_information_content < min_ancestor_information_content
        ):
            return None
        return TermPairwiseSimilarity(
            subject_id=subject,
            object_id=object,
            ancestor_id=(
                self.codec.concept_curie(similarity.ancestor_id)
                if similarity.ancestor_id is not None
                else None
            ),
            subject_information_content=similarity.information_content_a,
            object_information_content=similarity.information_content_b,
            ancestor_information_content=similarity.ancestor_information_content,
            jaccard_similarity=similarity.jaccard,
            phenodigm_score=similarity.phenodigm or None,
        )
