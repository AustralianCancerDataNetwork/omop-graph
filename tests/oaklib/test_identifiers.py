"""Identifiers: concept codes as CURIEs, unresolvable CURIEs and the prefix map."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest
from fixtures.cdm_builders import (
    classified,
    concept,
    edge_pair,
    relationship,
    relationship_class,
    sqlite_cdm,
)
from oaklib.datamodels.search import SearchConfiguration, SearchProperty
from oaklib.datamodels.vocabulary import HAS_DBXREF, IS_A, OWL_OBJECT_PROPERTY
from omop_alchemy.cdm.model.vocabulary.concept import Concept
from omop_alchemy.cdm.model.vocabulary.concept_ancestor import Concept_Ancestor
from omop_alchemy.cdm.model.vocabulary.concept_relationship import Concept_Relationship
from omop_alchemy.cdm.model.vocabulary.relationship import Relationship
from omop_alchemy.cdm.model.vocabulary.vocabulary import Vocabulary

from omop_graph.extensions.omop_alchemy import (
    HierarchyDirection,
    PredicateKind,
    RelationshipClass,
    RelationshipMapping,
)
from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

PARENT, CHILD, BRACES, BRACKETS, COLON, SPACE = range(400, 406)
VALID_DUPLICATE, RETIRED_DUPLICATE, AMBIGUOUS_ONE, AMBIGUOUS_TWO = range(410, 414)
MIXED_VALID_ONE, MIXED_VALID_TWO, MIXED_RETIRED_ONE, MIXED_RETIRED_TWO = range(414, 418)
SPACED_VOCABULARY, DASHED_VOCABULARY, SOURCE = 420, 421, 430
UNKNOWN = "SNOMED:nope"


def curie(concept_id: int) -> str:
    return f"OMOP:{concept_id}"


@pytest.fixture(scope="module")
def adapter() -> Iterator[OMOPAlchemyImplementation]:
    vocabularies = ("SNOMED", "UCUM", "Nebraska Lexicon", "DRG", "ICD10CM", "A B", "A-B")
    rows = [
        concept(PARENT),
        concept(CHILD),
        concept(BRACES, vocabulary_id="UCUM", concept_code="{tbl}"),
        concept(BRACKETS, vocabulary_id="UCUM", concept_code="[iU]/L"),
        concept(COLON, vocabulary_id="Nebraska Lexicon", concept_code="X:Y"),
        concept(SPACE, concept_code="with space"),
        concept(VALID_DUPLICATE, vocabulary_id="DRG", concept_code="014"),
        concept(RETIRED_DUPLICATE, vocabulary_id="DRG", concept_code="014", invalid_reason="D"),
        concept(AMBIGUOUS_ONE, vocabulary_id="DRG", concept_code="015"),
        concept(AMBIGUOUS_TWO, vocabulary_id="DRG", concept_code="015"),
        concept(MIXED_VALID_ONE, vocabulary_id="DRG", concept_code="016"),
        concept(MIXED_VALID_TWO, vocabulary_id="DRG", concept_code="016"),
        concept(MIXED_RETIRED_ONE, vocabulary_id="DRG", concept_code="016", invalid_reason="D"),
        concept(MIXED_RETIRED_TWO, vocabulary_id="DRG", concept_code="016", invalid_reason="U"),
        concept(SPACED_VOCABULARY, vocabulary_id="A B", concept_code="1"),
        concept(DASHED_VOCABULARY, vocabulary_id="A-B", concept_code="1"),
        concept(SOURCE, vocabulary_id="ICD10CM", concept_code="I48.9", standard_concept=None),
        *(
            Vocabulary(vocabulary_id=v, vocabulary_name=v, vocabulary_concept_id=i)
            for i, v in enumerate(vocabularies)
        ),
        relationship("Is a", "Subsumes"),
        relationship("Subsumes", "Is a"),
        relationship("Maps to", "Mapped from"),
        relationship("Mapped from", "Maps to"),
        relationship_class(PredicateKind.IDENTITY, "Exact equivalence"),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - up", HierarchyDirection.UP),
        relationship_class(PredicateKind.HIERARCHY, "Taxonomic - down", HierarchyDirection.DOWN),
        classified("Maps to", PredicateKind.IDENTITY, "Exact equivalence"),
        classified("Mapped from", PredicateKind.IDENTITY, "Exact equivalence"),
        classified("Is a", PredicateKind.HIERARCHY, "Taxonomic - up", generalises=True),
        classified("Subsumes", PredicateKind.HIERARCHY, "Taxonomic - down"),
        *edge_pair(CHILD, "Is a", "Subsumes", PARENT),
        *edge_pair(SOURCE, "Maps to", "Mapped from", PARENT),
    ]
    models = (
        Concept,
        Concept_Ancestor,
        Relationship,
        Concept_Relationship,
        Vocabulary,
        RelationshipClass,
        RelationshipMapping,
    )
    with sqlite_cdm(models, rows) as sessions:
        yield OMOPAlchemyImplementation(kg=KnowledgeGraph(sessions))


@pytest.mark.parametrize(
    ("concept_id", "vocabulary_id", "code", "expected"),
    [
        (BRACES, "UCUM", "{tbl}", "UCUM:%7Btbl%7D"),
        (BRACKETS, "UCUM", "[iU]/L", "UCUM:%5BiU%5D%2FL"),
        (COLON, "Nebraska Lexicon", "X:Y", "Nebraska_Lexicon:X%3AY"),
        (SPACE, "SNOMED", "with space", "SNOMED:with%20space"),
    ],
)
def test_a_concept_code_is_percent_encoded_and_resolves_back(
    adapter, concept_id: int, vocabulary_id: str, code: str, expected: str
) -> None:
    assert adapter.codec.code_curie(vocabulary_id, code) == expected
    assert adapter.codec.resolve_concept(expected) == concept_id


def test_every_concept_code_is_a_cross_reference_that_sssom_accepts(adapter) -> None:
    cross_references = {
        m.subject_id: m.object_id for m in adapter.sssom_mappings() if m.predicate_id == HAS_DBXREF
    }

    assert cross_references[curie(BRACES)] == "UCUM:%7Btbl%7D"
    assert cross_references[curie(COLON)] == "Nebraska_Lexicon:X%3AY"
    assert {m.object_id for m in adapter.sssom_mappings([curie(BRACKETS)])} == {"UCUM:%5BiU%5D%2FL"}


def test_a_code_is_looked_up_by_vocabulary_id_or_prefix(adapter) -> None:
    assert adapter.codec.resolve_concept("Nebraska Lexicon:X%3AY") == COLON
    assert adapter.codec.resolve_concept("Nebraska_Lexicon:X:Y") == COLON


@pytest.mark.parametrize(
    "unknown", [UNKNOWN, "OMOP:abc", "nothing", "SNOMED:", "OMOP_REL:Maps%20to"]
)
def test_a_curie_that_names_no_concept_is_not_resolved(adapter, unknown: str) -> None:
    assert adapter.codec.resolve_concept(unknown) is None


def test_unknown_curies_are_left_out_instead_of_raising(adapter) -> None:
    assert list(adapter.relationships(subjects=[UNKNOWN])) == []
    assert list(adapter.relationships(objects=["OMOP:abc"])) == []
    assert list(adapter.relationships(subjects=[UNKNOWN], include_entailed=True)) == []
    assert list(adapter.entailed_outgoing_relationships(UNKNOWN)) == []
    assert list(adapter.hierarchical_parents(UNKNOWN)) == []
    assert list(adapter.relationships_metadata([(UNKNOWN, IS_A, curie(PARENT))])) == [
        ((UNKNOWN, IS_A, curie(PARENT)), [])
    ]
    assert adapter.label(UNKNOWN) is None
    assert list(adapter.sssom_mappings([UNKNOWN])) == []
    assert list(adapter.simple_mappings_by_curie(UNKNOWN)) == []
    assert list(adapter.get_transitive_mappings_by_curie(UNKNOWN)) == []
    adapter.ancestor_graph(UNKNOWN)
    adapter.descendant_graph(UNKNOWN)
    adapter.node(UNKNOWN)


def test_a_valid_concept_is_preferred_over_a_retired_one_with_the_same_code(adapter) -> None:
    assert adapter.codec.resolve_concept("DRG:014") == VALID_DUPLICATE
    assert adapter.label("DRG:014") == f"Concept {VALID_DUPLICATE}"


def test_an_ambiguous_code_is_unresolved_and_the_warning_lists_the_concepts(
    adapter, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert adapter.codec.resolve_concept("DRG:015") is None
        assert adapter.label("DRG:015") is None

    assert f"Use one of {curie(AMBIGUOUS_ONE)}, {curie(AMBIGUOUS_TWO)}" in caplog.text


def test_a_search_by_an_ambiguous_code_yields_every_candidate(adapter) -> None:
    identifier = SearchConfiguration(properties=[SearchProperty.IDENTIFIER.text])

    assert set(adapter.basic_search("DRG:015", identifier)) == {
        curie(AMBIGUOUS_ONE),
        curie(AMBIGUOUS_TWO),
    }
    limited = SearchConfiguration(properties=[SearchProperty.IDENTIFIER.text], limit=1)
    assert len(list(adapter.basic_search("DRG:015", limited))) == 1


def test_a_search_by_a_code_shared_by_valid_and_retired_concepts_names_only_the_valid_ones(
    adapter,
) -> None:
    valid = {curie(MIXED_VALID_ONE), curie(MIXED_VALID_TWO)}
    for include_obsoletes in (False, True):
        configuration = SearchConfiguration(
            properties=[SearchProperty.IDENTIFIER.text],
            include_obsoletes_in_results=include_obsoletes,
        )
        assert set(adapter.basic_search("DRG:016", configuration)) == valid

    limited = SearchConfiguration(properties=[SearchProperty.IDENTIFIER.text], limit=1)
    assert len(list(adapter.basic_search("DRG:016", limited))) == 1


def test_is_a_is_known_by_both_of_its_predicate_curies(adapter) -> None:
    for predicate in (IS_A, "OMOP_REL:Is%20a"):
        assert list(adapter.owl_types([predicate])) == [(predicate, OWL_OBJECT_PROPERTY)]
        assert list(adapter.dangling([predicate])) == []
        assert adapter.label(predicate) == adapter.label(IS_A) is not None
        assert adapter.entity_metadata_map(predicate)["id"] == [predicate]


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (str(PARENT), PARENT),
        (curie(PARENT), PARENT),
        ("SNOMED:with%20space", SPACE),
        (UNKNOWN, None),
        ("nothing", None),
    ],
)
def test_a_reference_is_a_bare_id_or_a_curie(adapter, reference: str, expected: int | None) -> None:
    assert adapter.codec.resolve_reference(reference) == expected


def test_a_parent_that_names_no_concept_raises(adapter) -> None:
    assert adapter._parent_id(str(PARENT)) == PARENT
    with pytest.raises(ValueError, match="names no concept"):
        adapter._parent_id(UNKNOWN)


def test_a_prefix_shared_by_vocabularies_resolves_nothing_and_breaks_nothing(
    adapter, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        assert adapter.codec.resolve_concept("A_B:1") is None
    assert "A B, A-B" in caplog.text
    assert adapter.codec.resolve_concept("A B:1") == SPACED_VOCABULARY
    assert adapter.codec.resolve_concept("A-B:1") == DASHED_VOCABULARY
    assert adapter.codec.resolve_concept("DRG:014") == VALID_DUPLICATE


def test_a_predicate_outside_omop_is_ignored(adapter) -> None:
    part_of = "BFO:0000050"

    assert list(adapter.ancestors(curie(CHILD), predicates=[IS_A, part_of])) == list(
        adapter.ancestors(curie(CHILD), predicates=[IS_A])
    )
    assert curie(PARENT) in adapter.ancestors(curie(CHILD), predicates=[IS_A, part_of])
    assert list(adapter.ancestors(curie(CHILD), predicates=[part_of], reflexive=False)) == []
    assert list(adapter.relationships(subjects=[curie(CHILD)], predicates=[part_of])) == []
    assert curie(PARENT) in adapter.roots(predicates=[part_of])


@pytest.mark.parametrize("prefix", ["Nebraska Lexicon", "Nebraska_Lexicon"])
def test_id_prefixes_take_a_vocabulary_id_or_its_curie_prefix(adapter, prefix: str) -> None:
    assert list(adapter.roots(id_prefixes=[prefix])) == [curie(COLON)]


def test_a_vocabulary_code_gives_the_mappings_and_names_its_concept(adapter) -> None:
    assert set(adapter.simple_mappings_by_curie("ICD10CM:I48.9")) == set(
        adapter.simple_mappings_by_curie(curie(SOURCE))
    )
    assert ("oio:hasDbXref", "ICD10CM:I48.9") in adapter.simple_mappings_by_curie("ICD10CM:I48.9")
    (mapping,) = adapter.get_transitive_mappings_by_curie("ICD10CM:I48.9")
    assert mapping.subject_id == curie(SOURCE)


def test_the_prefix_map_covers_the_prefixes_the_adapter_emits(adapter) -> None:
    prefixes = adapter.prefix_map()

    assert prefixes["OMOP"] == "https://athena.ohdsi.org/search-terms/terms/"
    assert {"OMOP_REL", "rdfs", "skos", "oio", "owl"} <= prefixes.keys()
    assert (
        adapter.curie_to_uri("rdfs:subClassOf") == "http://www.w3.org/2000/01/rdf-schema#subClassOf"
    )
    assert adapter.curie_to_uri(curie(PARENT)) == f"{prefixes['OMOP']}{PARENT}"
    assert prefixes["ICD10CM"].startswith("https://")
    assert "Nebraska_Lexicon" not in prefixes
    assert adapter.curie_to_uri("Nebraska_Lexicon:X%3AY") is None


def test_a_predicate_that_matches_nothing_is_mentioned_once(
    adapter, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        list(adapter.ancestors(curie(CHILD), predicates=[IS_A, "BFO:0000051"]))
        list(adapter.ancestors(curie(CHILD), predicates=["BFO:0000051"]))

    assert caplog.text.count("The predicate BFO:0000051 names no OMOP relationship") == 1


def test_a_filter_that_names_no_concept_is_mentioned(
    adapter, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert list(adapter.relationships(subjects=[UNKNOWN])) == []

    assert f"No CURIE names a concept, so nothing matches: ['{UNKNOWN}']" in caplog.text
