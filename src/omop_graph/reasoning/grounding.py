from dataclasses import dataclass, field
from typing import Optional, Iterable, Generator, Union

from omop_graph.reasoning.resolvers import ResolverConfidence, ResolverPipeline
from omop_graph.graph.paths import find_standard_paths, StandardConcept
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.constraints import SearchConstraintConcept
from omop_graph.graph.edges import PredicateKind, HIERARCHICAL_PREDICATE_KINDS
from omop_graph.graph.scoring import StandardConceptWithScore, score_standard_concepts
from omop_graph.graph.paths import get_unique_standard_concepts

import logging
logger = logging.getLogger(__name__)


    
@dataclass(frozen=True)
class GroundingConstraints:
    parent_ids: Optional[tuple[int, ...]]
    search_constraint: Optional[SearchConstraintConcept]
    max_depth: int = 6
    predicate_kinds: frozenset[PredicateKind] = HIERARCHICAL_PREDICATE_KINDS


def ground_term(
    resolver_pipeline: ResolverPipeline,
    kg: KnowledgeGraph,
    text: str,
    constraints: GroundingConstraints,
    max_candidates: int = 10  # We later also only return a singular candidate
) -> list[StandardConceptWithScore]:

    standard_concepts: list[StandardConcept] = []

    # NOTE: Maybe not do this as a for-loop as we have so many calls to the DB this way! 
    # Probably could do it batched

    search_constraints = constraints.search_constraint
    if search_constraints is not None:
        search_constraints.check(kg)  # Check that the constraints are valid to return something

    resolved = resolver_pipeline.resolve(kg, text, constraints=search_constraints)
    resolved = list(resolved)
    for hit in resolved:
        if constraints.parent_ids is not None:
            # NOTE: Rename the function?
            candidate_standard_concepts = find_hierarchy_paths(
                kg,
                hit.concept_id,
                constraints.parent_ids,
                max_depth=constraints.max_depth,
                predicate_kinds=constraints.predicate_kinds,
                max_paths=5,
                resolver_confidence=hit.resolver_confidence,
            )
            # TODO: Filter exact smae concepts so we don't do the scoring multiple times?
            # This is important when we do embeddings as they are costly

            if not candidate_standard_concepts:
                concept_name = kg.concept_view(hit.concept_id).concept_name
                logger.debug(f"Failed hierarchy constraint (no path to parents): {hit.concept_id} ({concept_name}), Parents {constraints.parent_ids}")
                continue  # fails hierarchy constraint
            
            standard_concepts.extend(candidate_standard_concepts)
        else:
            raise NotImplementedError("Non parent_id is not supported")

    # Filter duplicates
    unique_standard_concepts = get_unique_standard_concepts(standard_concepts)

    # NOTE: This is temp! We should get the embeddings directly from the database for the standard concepts
    # The only one we need to calculate is the one from the input text
    from omop_spires.client.instructor_client import LLMClient
    embedding_client = LLMClient(
        model="qwen3-embedding:8b",
        api_base="http://ollama:11434/v1",
        api_key=''
    )

    # Combine the text with the standard_concepts
    batch = [text] + [sc.concept_name for sc in unique_standard_concepts]
    embeddings = embedding_client.embeddings(batch)

    input_embedding = embeddings[0:1]  # Shape (1, embedding_dim)
    standard_concept_embeddings = embeddings[1:] # Shape (num_concepts, embedding_dim)

    similarity_scores = embedding_client.cosine_similarity(input_embedding, standard_concept_embeddings)[0] # Shape (num_concepts,)

    ranked_standard_concepts = score_standard_concepts(
        standard_concepts=tuple(unique_standard_concepts),
        kg=kg,
        # NOTE: Could do these eventually
        similarity_scores=similarity_scores
    )

    ranked_standard_concepts.sort(key=lambda sc: sc.total_score, reverse=True)
    return ranked_standard_concepts
    
def find_hierarchy_paths(
    kg: KnowledgeGraph,
    concept_id: int,
    parent_ids: tuple[int, ...],
    *,
    max_depth: int,
    resolver_confidence: ResolverConfidence,
    max_paths: int = 3,
    predicate_kinds: frozenset[PredicateKind] = HIERARCHICAL_PREDICATE_KINDS,
    lowest_cost: Optional[float] = None,
) -> list[StandardConcept]:
    paths = []

    for parent in parent_ids:
        # found, trace = find_shortest_paths(
        found = find_standard_paths(
            kg,
            source=concept_id,
            target=parent,
            predicate_kinds=predicate_kinds,
            max_depth=max_depth,
            max_paths=max_paths,
            lowest_cost=lowest_cost,
            resolver_confidence=resolver_confidence,
        )
        paths.extend(found)

    return paths
