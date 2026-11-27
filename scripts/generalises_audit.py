"""Measure which relationships generalise through the ``Is a`` hierarchy.

For each attribute, composition and association relationship R, a sample of stored
edges ``(P, R, F)`` is taken where P has ``Is a`` children. For every child C of P the
script records whether C has R to F or to a descendant of F in the strict ``Is a``
hierarchy, and whether C has R at all. A relationship is proposed as generalising when
its children carry the parent's object and its reverse relationship does not.

The proposal is printed next to the flag in ``predicate_mapping.csv``. It measures
whether the data is consistent under inheritance, not whether the claim holds, so a
proposal is a prompt for review and not a replacement for it. A relationship whose
measurement or whose reverse's measurement timed out is reported as unmeasured, and its
flag is left as it is.

Run against the database of the active oa-configurator config, read-only::

    python scripts/generalises_audit.py --output generalises_audit.csv \\
        --provenance src/omop_graph/data/generalises_provenance.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import date
from enum import Enum
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from omop_alchemy.cdm.model import normalised_flag_expr
from omop_alchemy.cdm.model.vocabulary import Concept_Relationship, Vocabulary
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, aliased

from omop_graph.cli import packaged_predicate_csv_dir
from omop_graph.db.session import cdm_session_factory, resolve_cdm_database
from omop_graph.extensions.classification_data import (
    NON_GENERALISING_SUBKINDS,
    read_classification_files,
)
from omop_graph.extensions.omop_alchemy import PredicateKind
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.graph.queries import q_all_predicates

logger = logging.getLogger(__name__)

IS_A = "Is a"
MEASURED_KINDS = (PredicateKind.ATTRIBUTE, PredicateKind.COMPOSITION, PredicateKind.ASSOCIATION)
QUERY_CANCELED = "57014"


class Basis(str, Enum):
    """The children a share is taken over."""

    HOLDERS = "holders"
    ALL = "all"


class Proposal(str, Enum):
    """What the measurement says about a relationship."""

    GENERALISES = "true"
    DOES_NOT = "false"
    UNMEASURED = "unmeasured"


@dataclass(frozen=True, slots=True)
class Settings:
    """
    The thresholds that turn a measurement into a proposal.

    Parameters
    ----------
    basis : Basis
        The children a share is taken over.
    min_pairs : int
        Fewest pairs a measurement needs to count.
    min_share : float
        Smallest share of the relationship for it to generalise.
    max_reverse : float
        Largest share of a measured reverse for the relationship to still generalise.
    """

    basis: Basis = Basis.HOLDERS
    min_pairs: int = 30
    min_share: float = 0.7
    max_reverse: float = 0.15


@dataclass(frozen=True, slots=True)
class Measurement:
    """
    The inheritance of one relationship through ``Is a``.

    Parameters
    ----------
    pairs : int
        Child and parent-object pairs measured.
    holders : int
        Pairs whose child has the relationship at all.
    covered : int
        Pairs whose child has the relationship to the parent's object or a descendant.
    """

    pairs: int
    holders: int
    covered: int

    def share(self, basis: Basis) -> float | None:
        """Share of children carrying the parent's object, None when nothing was measured."""
        denominator = self.holders if basis is Basis.HOLDERS else self.pairs
        return self.covered / denominator if denominator else None


@dataclass(frozen=True, slots=True)
class TimedOut:
    """A measurement that ran past the statement timeout."""


def measure(
    session: Session, kg: KnowledgeGraph, relationship_id: str, sample_edges: int, sample_pairs: int
) -> Measurement:
    """
    Measure how the children of a parent inherit a relationship.

    Parameters
    ----------
    session : Session
        Read-only CDM session.
    kg : KnowledgeGraph
        Provides the strict ``Is a`` ancestors, the walk that entailment follows.
    relationship_id : str
        The relationship to measure.
    sample_edges : int
        Stored edges of the relationship sampled, taken only from parents with children
        and in storage order.
    sample_pairs : int
        Cap on the child and object pairs taken from those edges.

    Returns
    -------
    Measurement
    """
    edge, isa, child_edge, held = (aliased(Concept_Relationship) for _ in range(4))

    def valid(alias):
        return normalised_flag_expr(alias.invalid_reason).is_(None)

    has_children = sa.exists().where(
        isa.concept_id_2 == edge.concept_id_1, isa.relationship_id == IS_A, valid(isa)
    )
    edges = (
        sa.select(edge.concept_id_1.label("parent"), edge.concept_id_2.label("filler"))
        .where(
            edge.relationship_id == relationship_id,
            valid(edge),
            edge.concept_id_1 != edge.concept_id_2,
            has_children,
        )
        .limit(sample_edges)
        .subquery()
    )
    pairs = (
        sa.select(child_edge.concept_id_1.label("child"), edges.c.parent, edges.c.filler)
        .join(edges, child_edge.concept_id_2 == edges.c.parent)
        .where(child_edge.relationship_id == IS_A, valid(child_edge))
        .limit(sample_pairs)
        .subquery()
    )
    rows = session.execute(
        sa.select(pairs.c.child, pairs.c.parent, pairs.c.filler, held.concept_id_2.label("held"))
        .select_from(pairs)
        .outerjoin(
            held,
            sa.and_(
                held.concept_id_1 == pairs.c.child,
                held.relationship_id == relationship_id,
                valid(held),
            ),
        )
    ).all()
    held_by_pair: dict[tuple[int, int, int], set[int]] = defaultdict(set)
    for child, parent, filler, held_id in rows:
        held_by_pair[(child, parent, filler)]
        if held_id is not None:
            held_by_pair[(child, parent, filler)].add(held_id)
    ancestors = kg.closure(
        {object_id for objects in held_by_pair.values() for object_id in objects}
    )
    holders = [(pair, objects) for pair, objects in held_by_pair.items() if objects]
    covered = sum(
        any(pair[2] == object_id or pair[2] in ancestors[object_id] for object_id in objects)
        for pair, objects in holders
    )
    return Measurement(len(held_by_pair), len(holders), covered)


def measure_within(
    session: Session,
    kg: KnowledgeGraph,
    relationship_id: str,
    sample_edges: int,
    sample_pairs: int,
    timeout: int,
) -> Measurement | TimedOut:
    """
    Measure a relationship, giving up after ``timeout`` seconds.

    Parameters
    ----------
    session : Session
        Read-only CDM session.
    kg : KnowledgeGraph
        As for :func:`measure`.
    relationship_id : str
        The relationship to measure.
    sample_edges : int
        As for :func:`measure`.
    sample_pairs : int
        As for :func:`measure`.
    timeout : int
        Seconds a PostgreSQL statement may run, ignored on other databases.

    Returns
    -------
    Measurement | TimedOut
        TimedOut when PostgreSQL cancelled a statement. Any other database error is raised.
    """
    if session.get_bind().dialect.name == "postgresql":
        session.execute(sa.text(f"SET statement_timeout = {int(timeout) * 1000}"))
    try:
        return measure(session, kg, relationship_id, sample_edges, sample_pairs)
    except DBAPIError as error:
        sqlstate = getattr(error.orig, "sqlstate", None) or getattr(error.orig, "pgcode", None)
        if sqlstate != QUERY_CANCELED:
            raise
        session.rollback()
        return TimedOut()


def propose(
    forward: Measurement | TimedOut, reverse: Measurement | TimedOut | None, settings: Settings
) -> Proposal:
    """
    Propose whether a relationship generalises from its own and its reverse's measurement.

    Parameters
    ----------
    forward : Measurement | TimedOut
        The relationship.
    reverse : Measurement | TimedOut | None
        Its reverse relationship, None when it has no classification and was not measured.
    settings : Settings
        The thresholds.

    Returns
    -------
    Proposal
        UNMEASURED when either measurement timed out. GENERALISES when the children carry
        the parent's object and the reverse does not. DOES_NOT when the data is too thin,
        too mixed, or both sides inherit.
    """
    if isinstance(forward, TimedOut) or isinstance(reverse, TimedOut):
        return Proposal.UNMEASURED
    share = forward.share(settings.basis)
    if forward.pairs < settings.min_pairs or share is None or share < settings.min_share:
        return Proposal.DOES_NOT
    reverse_share = (
        reverse.share(settings.basis) if reverse and reverse.pairs >= settings.min_pairs else None
    )
    if reverse_share is None or reverse_share <= settings.max_reverse:
        return Proposal.GENERALISES
    return Proposal.DOES_NOT


def sensitivity(
    measured: dict[str, Measurement | TimedOut],
    reverse_of: dict[str, str | None],
    wanted: list[str],
    base: Settings,
) -> list[dict]:
    """
    Count the proposals that change when one threshold moves to a nearby value.

    Parameters
    ----------
    measured : dict[str, Measurement | TimedOut]
        The measurement of every relationship, reverses included.
    reverse_of : dict[str, str | None]
        The reverse relationship id of each relationship.
    wanted : list[str]
        The relationships proposed for.
    base : Settings
        The thresholds the proposals are compared with.

    Returns
    -------
    list[dict]
        One entry per nearby threshold with its name, value, the number of changed
        proposals and their relationship ids.
    """
    nearby = {
        "min_share": (0.6, 0.8),
        "min_pairs": (20, 50),
        "max_reverse": (0.10, 0.25),
    }

    def proposals(settings: Settings) -> dict[str, Proposal]:
        return {
            rid: propose(measured[rid], measured.get(reverse_of.get(rid) or ""), settings)
            for rid in wanted
        }

    reference = proposals(base)
    results = []
    for name, values in nearby.items():
        for value in values:
            changed = sorted(
                rid
                for rid, proposal in proposals(replace(base, **{name: value})).items()
                if proposal != reference[rid]
            )
            results.append(
                {
                    "threshold": name,
                    "value": value,
                    "changed": len(changed),
                    "relationships": changed,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    defaults = Settings()
    parser.add_argument("--output", help="CSV file for the results")
    parser.add_argument(
        "--provenance", help="JSON file for the release, date, parameters and sensitivity"
    )
    parser.add_argument(
        "--classification-dir",
        type=Path,
        default=packaged_predicate_csv_dir(),
        help="Directory of the classification files holding the current flags (default: packaged)",
    )
    parser.add_argument(
        "--relationship", action="append", help="Measure only these relationship ids"
    )
    parser.add_argument(
        "--basis",
        type=Basis,
        choices=list(Basis),
        default=defaults.basis,
        metavar="{" + ",".join(basis.value for basis in Basis) + "}",
        help="Take the share over the children holding the relationship, or over all children (default: %(default)s)",
    )
    parser.add_argument("--sample-edges", type=int, default=1500)
    parser.add_argument("--sample-pairs", type=int, default=15000)
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per statement")
    parser.add_argument("--min-pairs", type=int, default=defaults.min_pairs)
    parser.add_argument("--min-share", type=float, default=defaults.min_share)
    parser.add_argument("--max-reverse", type=float, default=defaults.max_reverse)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings(args.basis, args.min_pairs, args.min_share, args.max_reverse)

    _, classification = read_classification_files(args.classification_dir)
    classified = classification.set_index("relationship_id")
    pairs = zip(classified["predicate_kind"], classified["predicate_subkind"])
    measurable = {
        rid
        for rid, (kind, subkind) in zip(classified.index, pairs)
        if PredicateKind(kind) in MEASURED_KINDS
        and (kind, subkind) not in NON_GENERALISING_SUBKINDS
    }
    sessions = cdm_session_factory(resolved=resolve_cdm_database())
    kg = KnowledgeGraph(sessions)
    with sessions() as session:
        release = session.scalar(
            sa.select(Vocabulary.vocabulary_version).where(Vocabulary.vocabulary_id == "None")
        )
        reverse_of = {
            row.relationship_id: row.reverse_relationship_id
            for row in session.execute(q_all_predicates())
        }
        wanted = sorted(set(args.relationship) & measurable if args.relationship else measurable)
        edge_counts = dict(
            session.execute(
                sa.select(Concept_Relationship.relationship_id, sa.func.count())
                .where(Concept_Relationship.is_valid_expr())
                .group_by(Concept_Relationship.relationship_id)
            ).all()
        )
        measured: dict[str, Measurement | TimedOut] = {}

        def cached(rid: str) -> Measurement | TimedOut:
            if rid not in measured:
                measured[rid] = (
                    measure_within(
                        session, kg, rid, args.sample_edges, args.sample_pairs, args.timeout
                    )
                    if edge_counts.get(rid)
                    else Measurement(0, 0, 0)
                )
            return measured[rid]

        rows = []
        for rid in wanted:
            forward = cached(rid)
            reverse_id = reverse_of.get(rid)
            reverse = cached(reverse_id) if reverse_id in classified.index else None
            proposal = propose(forward, reverse, settings)
            current = bool(classified.loc[rid, "generalises"])
            rows.append(
                {
                    "relationship_id": rid,
                    "edges": edge_counts.get(rid, 0),
                    "pairs": getattr(forward, "pairs", None),
                    "share_of_holders": None
                    if isinstance(forward, TimedOut)
                    else forward.share(Basis.HOLDERS),
                    "share_of_all": None
                    if isinstance(forward, TimedOut)
                    else forward.share(Basis.ALL),
                    "reverse_id": reverse_id,
                    "reverse_pairs": getattr(reverse, "pairs", None),
                    "reverse_share": reverse.share(settings.basis)
                    if isinstance(reverse, Measurement)
                    else None,
                    "proposed": proposal.value,
                    "current": current,
                    "differs": proposal is not Proposal.UNMEASURED
                    and (proposal is Proposal.GENERALISES) != current,
                    "timed_out": isinstance(forward, TimedOut),
                    "reverse_timed_out": isinstance(reverse, TimedOut),
                }
            )
            logger.info("%s: %s, current %s", rid, proposal.value, current)

    result = pd.DataFrame(rows)
    if args.output:
        result.to_csv(args.output, index=False)
    if args.provenance:
        flips = sensitivity(measured, reverse_of, wanted, settings)
        provenance = {
            "vocabulary_version": release,
            "date": date.today().isoformat(),
            "settings": {**asdict(settings), "basis": settings.basis.value},
            "sample_edges": args.sample_edges,
            "sample_pairs": args.sample_pairs,
            "sample_order": "storage order, no ordering",
            "timeout_seconds": args.timeout,
            "measured": len(result),
            "proposals": result["proposed"].value_counts().to_dict(),
            "timed_out": sorted(
                result.loc[result["timed_out"] | result["reverse_timed_out"], "relationship_id"]
            ),
            "differs_from_csv": sorted(result.loc[result["differs"], "relationship_id"]),
            "sensitivity": flips,
        }
        Path(args.provenance).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    logger.info("%d measured, %d differ from the CSV", len(result), int(result["differs"].sum()))


if __name__ == "__main__":
    main()
