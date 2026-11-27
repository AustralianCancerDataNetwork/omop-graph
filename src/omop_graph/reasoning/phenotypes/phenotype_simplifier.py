from collections import defaultdict
from dataclasses import dataclass, field

from omop_graph.extensions.relationship_classification import HierarchyDirection
from omop_graph.graph.kg import KnowledgeGraph


@dataclass
class ParentStatistics:
    descendants: set[int] = field(default_factory=set)
    found: set[int] = field(default_factory=set)
    coverage: int = 0
    pollution: int = 0
    completeness: float = 0.0
    purity: float = 0.0
    max_depth: int = 0


def find_common_parents(
    seeds: list[int],
    kg: KnowledgeGraph,
    min_coverage: int = 2,
    max_up_depth: int | None = None,  # optional safety valve
) -> dict[int, ParentStatistics]:
    """
    Find the parents that cover several seeds and how well each one fits them.

    A parent is a standard concept above the seeds in ``concept_ancestor``. The concepts
    below a parent in the strict ``Is a`` hierarchy are its descendants, and those that are
    neither a seed nor below one pollute it.

    Parameters
    ----------
    seeds : list[int]
        The concepts to find parents for.
    kg : KnowledgeGraph
        The graph to read.
    min_coverage : int
        Keep the parents that cover at least this many seeds.
    max_up_depth : int, optional
        How many levels above the seeds to look. Unlimited when not given.

    Returns
    -------
    dict[int, ParentStatistics]
        The statistics of each parent, by its concept id.

    Raises
    ------
    ValueError
        If a seed or a parent maps to several standard concepts.
    UnknownConceptError
        If a seed does not exist.
    """
    seed_set = set(seeds)
    standard_seeds = {
        standard_id
        for standard_id in kg.standard_concept(seed_set).values()
        if standard_id is not None
    }
    candidates: defaultdict[int, ParentStatistics] = defaultdict(ParentStatistics)
    chosen = seed_set | standard_seeds
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

        for parent in kg.parents(current):
            # record evidence
            candidates[parent].found.add(origin)
            candidates[parent].descendants.add(origin)
            candidates[parent].max_depth = max(candidates[parent].max_depth, depth + 1)

            frontier.append((parent, origin, depth + 1))

    standard_map = kg.standard_concept(candidates.keys())

    final: defaultdict[int, ParentStatistics] = defaultdict(ParentStatistics)
    for parent, stats in candidates.items():
        std_parent = standard_map[parent] or parent
        final[std_parent].descendants |= stats.descendants
        final[std_parent].found |= stats.found

        final[std_parent].max_depth = max(final[std_parent].max_depth, stats.max_depth)

    for stats in final.values():
        stats.coverage = len(stats.descendants)

    below_parents = kg.closure(final, HierarchyDirection.DOWN)
    covered_by_seeds = chosen | set().union(*kg.closure(chosen, HierarchyDirection.DOWN).values())
    for parent, stats in final.items():
        all_desc = below_parents[parent]
        stats.pollution = len(all_desc - covered_by_seeds)
        stats.descendants |= all_desc
        denom = stats.coverage + stats.pollution
        stats.purity = stats.coverage / denom if denom else 0.0
        stats.completeness = stats.coverage / max(len(seeds), 1)

    return {parent: stats for parent, stats in final.items() if stats.coverage >= min_coverage}


def greedy_parent_cover(
    seeds: set[int],
    candidates: dict[int, ParentStatistics],
    *,
    target_coverage_ratio: float = 1.0,  # e.g. 0.95 for “good enough”
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.3,
    delta: float = 0.7,
    min_gain: int = 1,
) -> list[int]:
    remaining = set(seeds)
    selected: list[int] = []

    def score(c: ParentStatistics, gain: int) -> float:
        if gain <= 0:
            return -1.0
        return (
            (gain**alpha)
            * (c.purity**beta)
            / ((1 + c.pollution) ** gamma * (1 + c.max_depth) ** delta)
        )

    while remaining:
        covered = len(seeds) - len(remaining)
        if covered / max(1, len(seeds)) >= target_coverage_ratio:
            break

        best_id = None
        best_score = -1.0
        best_gain_set: set[int] = set()

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
            if c1 == c2:
                continue

            if g1.found <= g2.found:
                relations.append(
                    {
                        "type": "subsumed_by",
                        "from": c1,
                        "to": c2,
                        "overlap": len(g1.found),
                    }
                )

    return relations
