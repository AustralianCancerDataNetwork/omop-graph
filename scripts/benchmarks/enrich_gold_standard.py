"""
Backfill `domain`/`parent_ids` for a gold-standard cases file from the real CDM hierarchy
(https://github.com/OHDSI/Ariadne/blob/main/data/gold_standards/exact_matching_gs.csv)

`GroundingConstraints` requires `parent_ids` as a hierarchy anchor; gold-standard sets sourced
from external mapping tools (e.g. OHDSI's text->concept benchmarks) don't carry one. This script
derives anchors per case from `concept_ancestor`/`concept`, at every requested hierarchy depth:

- `domain` is set to the target concept's own `domain_id`.
- `parent_ids_by_level["N"]` is the target's standard, same-domain ancestors at exactly
  `min_levels_of_separation == N`, for each `N` in `--levels` (default 1-5). If none exist in
  the same domain, falls back to any-domain standard ancestors at that level (a real graph edge,
  just outside the target's own domain -- e.g. SNOMED finding/disorder pairs). If the hierarchy
  doesn't reach that deep at all, the level's list is `[]` -- a legitimate "no anchor exists at
  this depth" outcome for the sensitivity sweep, not an error.
- `parent_ids` is kept as an alias for level 1, for backward compatibility with every script and
  cases file that expects a flat `parent_ids` field.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from sqlalchemy import text

from omop_graph.db.session import make_engine

DOMAIN_QUERY = text(
    """
    SELECT concept_id, domain_id
    FROM omop.concept
    WHERE concept_id = ANY(:ids)
    """
)

ANCESTOR_QUERY = text(
    """
    SELECT ca.descendant_concept_id, ca.ancestor_concept_id, ca.min_levels_of_separation,
           c.concept_name, c.domain_id
    FROM omop.concept_ancestor ca
    JOIN omop.concept c ON c.concept_id = ca.ancestor_concept_id
    WHERE ca.descendant_concept_id = ANY(:ids)
      AND ca.min_levels_of_separation > 0
      AND c.standard_concept = 'S'
    """
)

@dataclass
class LevelStats:
    empty: int = 0
    cross_domain_fallback: int = 0
    single: int = 0
    multiple: int = 0

    def __iadd__(self, other: "LevelStats") -> "LevelStats":
        self.empty += other.empty
        self.cross_domain_fallback += other.cross_domain_fallback
        self.single += other.single
        self.multiple += other.multiple
        return self

@dataclass
class Stats:
    levels: Dict[int, LevelStats]
    no_target: int = 0

    @classmethod
    def empty_for(cls, levels: List[int]) -> "Stats":
        return cls(levels={level: LevelStats() for level in levels})

    def __iadd__(self, other: "Stats") -> "Stats":
        for level, level_stats in other.levels.items():
            self.levels.setdefault(level, LevelStats())
            self.levels[level] += level_stats
        self.no_target += other.no_target
        return self


def enrich_cases(engine, cases: List[Dict], levels: List[int]) -> Stats:
    target_ids = sorted({c["expected_concept_id"] for c in cases if c.get("expected_concept_id")})

    with engine.connect() as conn:
        domain_by_target = {
            row.concept_id: row.domain_id
            for row in conn.execute(DOMAIN_QUERY, {"ids": target_ids}).fetchall()
        }

        ancestors_by_target: Dict[int, List] = defaultdict(list)
        for row in conn.execute(ANCESTOR_QUERY, {"ids": target_ids}).fetchall():
            ancestors_by_target[row.descendant_concept_id].append(row)

    stats: Stats = Stats(levels={level: LevelStats() for level in levels}, no_target=0)

    for case_ in cases:
        target_id = case_.get("expected_concept_id")
        if not target_id:
            stats.no_target += 1
            continue

        target_domain = domain_by_target.get(target_id)
        case_["domain"] = target_domain
        all_ancestors = ancestors_by_target.get(target_id, [])

        parent_ids_by_level: Dict[str, List[int]] = {}
        for level in levels:
            at_level = [a for a in all_ancestors if a.min_levels_of_separation == level]
            same_domain = [a for a in at_level if a.domain_id == target_domain]

            level_stats: LevelStats = stats.levels[level]
            if same_domain:
                chosen = same_domain
            elif at_level:
                chosen = at_level
                level_stats.cross_domain_fallback += 1
            else:
                chosen = []
                level_stats.empty += 1

            ids = sorted({a.ancestor_concept_id for a in chosen})
            parent_ids_by_level[str(level)] = ids
            if len(ids) == 1:
                level_stats.single += 1
            elif len(ids) > 1:
                level_stats.multiple += 1

        case_["parent_ids_by_level"] = parent_ids_by_level
        case_["parent_ids"] = parent_ids_by_level.get("1", [])
        case_.pop("broader_parent_id_candidates", None)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases_file", type=Path)
    parser.add_argument(
        "--levels",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated hierarchy depths (min_levels_of_separation) to derive anchors for.",
    )
    args = parser.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    payload = json.loads(args.cases_file.read_text())
    engine = make_engine()

    total_stats = Stats.empty_for(levels)
    for bucket_name, cases in payload.items():
        stats = enrich_cases(engine, cases, levels)
        print(f"[{bucket_name}] {stats}")
        total_stats += stats

    print(f"[total] {total_stats}")

    args.cases_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {args.cases_file}")


if __name__ == "__main__":
    main()
