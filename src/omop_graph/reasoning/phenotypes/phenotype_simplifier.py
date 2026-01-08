from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Dict, Set, List
from omop_graph.graph.kg import KnowledgeGraph
from ..concept_handlers import standardise_ids

@dataclass 
class ParentStatistics: 
    descendants: set[int] = field(default_factory=set)
    found: set[int] = field(default_factory=set)
    coverage: int = 0 
    pollution: int = 0 
    completeness: float = 0.0 
    purity: float = 0.0 
    max_depth: int = 0


def parent_search(
    kg: KnowledgeGraph,
    concept_id: int,
) -> set[int]:
    """
    One-hop parents using ONLY 'Is a'
    """
    return {
        e.object_id
        for e in kg.iter_edges(
            concept_id,
            direction="out",
            predicate="Is a",
        )
        if e.object_id != concept_id
    }

def descendants_exhaustive_subsumes(
    kg: KnowledgeGraph,
    root_id: int,
    exclude_roots: set[int] | None = None,
) -> set[int]:
    """
    Exhaustive descendant closure using ONLY 'Subsumes'
    """
    
    if exclude_roots is None:
        exclude_roots = set()
        
    descendants: set[int] = set()
    frontier: list[int] = [root_id]

    while frontier:
        current = frontier.pop()
        for e in kg.iter_edges(
            current,
            direction="out",
            predicate="Subsumes",
        ):
            child = e.object_id
            if child == current or child in descendants:
                continue
            descendants.add(child)
            if child in exclude_roots:
                continue
            frontier.append(child)

    return descendants

def find_common_parents(
    seeds: list[int],
    kg: KnowledgeGraph,
    min_coverage: int = 2,
    max_up_depth: int | None = None,  # optional safety valve
) -> dict[int, ParentStatistics]:

    seed_set = set(seeds)
    standard_seeds = set(standardise_ids(seed_set, kg))
    candidates: defaultdict[int, ParentStatistics] = defaultdict(ParentStatistics)
    exclude = seed_set | standard_seeds
    # frontier items: (current_concept, originating_seed, depth)
    frontier: list[tuple[int, int, int]] = [(s, s, 0) for s in seeds]
    visited: set[tuple[int, int]] = set()
    
    while frontier:
        current, origin, depth = frontier.pop(0)

        if (current, origin) in visited:
            continue
        visited.add((current, origin))

        if max_up_depth is not None and depth >= max_up_depth:
            continue

        for parent in parent_search(kg, current):
            # record evidence
            candidates[parent].found.add(origin)
            candidates[parent].descendants.add(origin)
            candidates[parent].max_depth = max(
                candidates[parent].max_depth, depth + 1
            )

            frontier.append((parent, origin, depth + 1))

    standard_map = standardise_ids(set(candidates.keys()), kg)

    final: defaultdict[int, ParentStatistics] = defaultdict(ParentStatistics)
    for parent, stats in candidates.items():
        std_parent = standard_map[parent]
        final[std_parent].descendants |= stats.descendants
        final[std_parent].found |= stats.found

        final[std_parent].max_depth = max(
            final[std_parent].max_depth, stats.max_depth
        )

    for stats in final.values():
        stats.coverage = len(stats.descendants)

    for parent, stats in final.items():
        all_desc = descendants_exhaustive_subsumes(kg, parent, exclude)
        stats.pollution = len(all_desc - standard_seeds - seed_set)
        final[parent].descendants |= all_desc
        denom = stats.coverage + stats.pollution
        stats.purity = stats.coverage / denom if denom else 0.0
        stats.completeness = stats.coverage / max(len(seeds), 1)

    return {
        parent: stats
        for parent, stats in final.items()
        if stats.coverage >= min_coverage
    }

def greedy_parent_cover(
    seeds: Set[int],
    candidates: Dict[int, ParentStatistics],
    *,
    target_coverage_ratio: float = 1.0,  # e.g. 0.95 for “good enough”
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.3,
    delta: float = 0.7,
    min_gain: int = 1,
) -> List[int]:
    remaining = set(seeds)
    selected: List[int] = []

    def score(c: ParentStatistics, gain: int) -> float:
        if gain <= 0:
            return -1.0
        return (gain ** alpha) * (c.purity ** beta) / ((1 + c.pollution) ** gamma * (1 + c.max_depth) ** delta)

    while remaining:
        covered = len(seeds) - len(remaining)
        if covered / max(1, len(seeds)) >= target_coverage_ratio:
            break

        best_id = None
        best_score = -1.0
        best_gain_set: Set[int] = set()

        for cid, c in candidates.items():
            gain_set = c.found & remaining
            gain = len(gain_set)
            if gain < min_gain:
                continue

            s = score(c, gain)
            if s > best_score:
                best_score = s
                best_id = cid
                best_gain_set = gain_set

        if best_id is None:
            # cannot make progress under constraints
            break

        selected.append(best_id)
        remaining -= best_gain_set

    return selected

def relate_groups(groups: dict[int, ParentStatistics]) -> list[dict]:
    relations = []

    for c1, g1 in groups.items():
        for c2, g2 in groups.items():
            if c1==c2:
                continue

            if g1.found <= g2.found:
                relations.append({
                    "type": "subsumed_by",
                    "from": c1,
                    "to": c2,
                    "overlap": len(g1.found),
                })

    return relations