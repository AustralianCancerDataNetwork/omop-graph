import pandas as pd
import pytest

from omop_graph.extensions.classification_data import (
    ClassificationDataError,
    check_classification,
    check_mapping_against_cdm,
    parse_generalises,
    read_classification_files,
)
from omop_graph.extensions.omop_alchemy import HierarchyDirection, PredicateKind

RELATIONSHIPS = {
    "Maps to": ("Maps to", "Maps to"),
    "Is a": ("Is a", "Subsumes"),
    "Subsumes": ("Subsumes", "Is a"),
    "Has scale type": ("Has scale type", "Scale type of"),
    "Scale type of": ("Scale type of", "Has scale type"),
    "Self": ("Self", "Self"),
    "Has finding site": ("Has finding site", "Finding site of"),
    "Finding site of": ("Finding site of", "Has finding site"),
}
HIERARCHY, IDENTITY_KIND = PredicateKind.HIERARCHY.value, PredicateKind.IDENTITY.value
UP, DOWN, IDENTITY, SITE = (
    (HIERARCHY, "Taxonomic", HierarchyDirection.UP.value),
    (HIERARCHY, "Taxonomic", HierarchyDirection.DOWN.value),
    (IDENTITY_KIND, "x", None),
    (PredicateKind.ATTRIBUTE.value, "Anatomical", None),
)
CLASSIFICATION = pd.DataFrame(
    [(HIERARCHY, "Taxonomic", "d", "s", "i"), (IDENTITY_KIND, "x", "d", "s", "i")],
    columns=["predicate_kind", "predicate_subkind", "description", "semantics", "inference"],
)


def _mapping(
    *rows: tuple[str, tuple[str, str, str | None]], generalising: tuple[str, ...] = ()
) -> pd.DataFrame:
    records = [(rid, RELATIONSHIPS[rid][0], *kind, rid in generalising) for rid, kind in rows]
    return pd.DataFrame(
        records,
        columns=[
            "relationship_id",
            "r_name",
            "predicate_kind",
            "predicate_subkind",
            "direction",
            "generalises",
        ],
    )


def _check(mapping: pd.DataFrame, **kwargs):
    return check_mapping_against_cdm(mapping, RELATIONSHIPS, **kwargs)


GOOD = _mapping(
    ("Maps to", IDENTITY),
    ("Is a", UP),
    ("Subsumes", DOWN),
    ("Has scale type", UP),
    ("Scale type of", DOWN),
    ("Self", IDENTITY),
    ("Has finding site", SITE),
    ("Finding site of", SITE),
    generalising=("Is a", "Has finding site"),
)


def test_a_consistent_mapping_passes_with_an_empty_report():
    report = _check(GOOD, strict=True)

    assert report.warnings == []


def test_a_name_differing_from_the_cdm_is_rejected():
    mapping = GOOD.copy()
    mapping.loc[1, "r_name"] = "Is a (SNOMED)"

    with pytest.raises(ClassificationDataError, match="relationship_name is 'Is a'"):
        _check(mapping)


def test_a_relationship_mapped_twice_is_rejected():
    with pytest.raises(ClassificationDataError, match="more than once.*Is a"):
        _check(pd.concat([GOOD, GOOD.iloc[[1]]]))


@pytest.mark.parametrize("strict", [False, True])
def test_unclassified_cdm_relationships_warn_and_fail_only_under_strict(strict):
    mapping = GOOD.iloc[1:]

    if strict:
        with pytest.raises(ClassificationDataError, match="1 CDM relationships have no.*'Maps to'"):
            _check(mapping, strict=True)
    else:
        assert _check(mapping).unclassified == ["Maps to"]


@pytest.mark.parametrize("strict", [False, True])
def test_classified_ids_missing_from_the_cdm_warn_and_fail_only_under_strict(strict):
    mapping = pd.concat(
        [GOOD, _mapping(("Maps to", IDENTITY)).assign(relationship_id="Subset only", r_name="x")]
    )

    if strict:
        with pytest.raises(ClassificationDataError, match="not in the CDM.*'Subset only'"):
            _check(mapping, strict=True)
    else:
        assert _check(mapping).unknown_ids == ["Subset only"]


@pytest.mark.parametrize(
    "mapping",
    [
        _mapping(("Maps to", IDENTITY), ("Is a", UP), ("Subsumes", UP)),
        _mapping(("Maps to", IDENTITY), ("Subsumes", DOWN), ("Is a", DOWN)),
    ],
)
def test_a_hierarchy_pair_with_one_direction_is_rejected_once(mapping):
    with pytest.raises(ClassificationDataError, match="are both") as error:
        _check(mapping)

    assert str(error.value).count("are both") == 1


def test_a_relationship_and_its_reverse_may_not_both_generalise():
    mapping = _mapping(
        ("Has finding site", SITE),
        ("Finding site of", SITE),
        generalising=("Has finding site", "Finding site of"),
    )

    with pytest.raises(ClassificationDataError, match="both generalise") as error:
        _check(mapping)

    assert str(error.value).count("both generalise") == 1


@pytest.mark.parametrize(
    ("hierarchy_id", "other_id"),
    [("Is a", "Subsumes"), ("Subsumes", "Is a"), ("Has scale type", "Scale type of")],
)
def test_a_pair_with_one_hierarchy_side_is_rejected_whichever_side_it_is(hierarchy_id, other_id):
    direction = UP if hierarchy_id in {"Is a", "Has scale type"} else DOWN
    mapping = _mapping((hierarchy_id, direction), (other_id, IDENTITY))

    with pytest.raises(
        ClassificationDataError,
        match=f"'{hierarchy_id}' is Hierarchy but its reverse '{other_id}' is Identity",
    ):
        _check(mapping)


def test_a_hierarchy_relationship_that_is_its_own_reverse_is_rejected():
    mapping = _mapping(("Maps to", IDENTITY), ("Self", UP))

    with pytest.raises(ClassificationDataError, match="'Self' is Hierarchy but is its own reverse"):
        _check(mapping)


def test_an_empty_relationship_vocabulary_fails_clearly():
    with pytest.raises(RuntimeError, match="relationship vocabulary is empty"):
        check_mapping_against_cdm(GOOD, {})


def test_a_mapping_without_a_single_cdm_id_fails_clearly():
    with pytest.raises(RuntimeError, match="None of the relationship mappings"):
        check_mapping_against_cdm(GOOD, {"Other": ("Other", "Other")})


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [(HIERARCHY, "x", "a"), (HIERARCHY, "x", "b")],
            "defined 2 times, differing in .*description",
        ),
        ([(HIERARCHY, "x", "a"), (HIERARCHY, "x", "a")], "defined 2 times$"),
        ([("Unknown", "x", "a")], "unknown classes: \\['Unknown'\\]"),
    ],
)
def test_a_classification_must_define_each_kind_and_subkind_once(rows, message):
    classification = pd.DataFrame(
        [(kind, subkind, description, "s", "i") for kind, subkind, description in rows],
        columns=["predicate_kind", "predicate_subkind", "description", "semantics", "inference"],
    )

    with pytest.raises(ClassificationDataError, match=message):
        check_classification(classification)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["", "false", "FALSE", " False "], [False] * 4),
        (["true", "True", " TRUE "], [True] * 3),
        ([None, True, False], [False, True, False]),
    ],
)
def test_generalises_flags_are_read_as_booleans(flags, expected):
    assert parse_generalises(pd.Series(flags, dtype="object")).tolist() == expected


@pytest.mark.parametrize("flag", ["yes", "1", "maybe"])
def test_a_generalises_flag_must_be_empty_true_or_false(flag):
    with pytest.raises(ClassificationDataError, match=f"generalises must be.*'{flag}'"):
        parse_generalises(pd.Series([flag, "true"]))


def _write(directory, classification: str, mapping: str):
    (directory / "predicate_classification.csv").write_text(
        "class,subclass,description,semantics,inference\n" + classification
    )
    (directory / "predicate_mapping.csv").write_text(mapping)


def test_read_classification_files_reads_and_checks_both_files(tmp_path):
    _write(
        tmp_path,
        "Hierarchy,Taxonomic,d,s,i\nIdentity,x,d,s,i\n",
        "class,subclass,r_id,r_name,direction\nHierarchy,Taxonomic,Is a,Is a,up\nIdentity,x,Maps to,Maps to,\n",
    )

    classification, mapping = read_classification_files(tmp_path)

    assert classification["predicate_subkind"].tolist() == ["Taxonomic", "x"]
    assert mapping["relationship_id"].tolist() == ["Is a", "Maps to"]
    assert mapping["direction"].tolist()[0] == "up"
    assert pd.isna(mapping["direction"].tolist()[1])


def test_a_mapping_without_a_direction_column_has_no_direction(tmp_path):
    _write(
        tmp_path, "Identity,x,d,s,i\n", "class,subclass,r_id,r_name\nIdentity,x,Maps to,Maps to\n"
    )

    _, mapping = read_classification_files(tmp_path)

    assert mapping["direction"].isna().all()


def test_the_generalises_column_is_read_and_defaults_to_false(tmp_path):
    _write(
        tmp_path,
        "Attribute,Anatomical,d,s,i\n",
        "class,subclass,r_id,r_name,generalises\n"
        "Attribute,Anatomical,Has finding site,Has finding site,true\n"
        "Attribute,Anatomical,Finding site of,Finding site of,\n",
    )

    _, mapping = read_classification_files(tmp_path)

    assert mapping["generalises"].tolist() == [True, False]


@pytest.mark.parametrize(
    ("kind", "subkind", "direction", "valid"),
    [
        ("Attribute", "Anatomical", "", True),
        ("Composition", "Physical", "", True),
        ("Association", "Therapeutic", "", True),
        ("Hierarchy", "Taxonomic", "up", True),
        ("Hierarchy", "Taxonomic", "down", False),
        ("Identity", "Exact equivalence", "", False),
        ("Metadata", "CDM version", "", False),
        ("Association", "Mapping", "", False),
        ("Association", "Administrative", "", False),
        ("Association", "Contraindication", "", False),
        ("Association", "Value mapping", "", False),
    ],
)
def test_only_some_relationships_may_generalise(tmp_path, kind, subkind, direction, valid):
    _write(
        tmp_path,
        f"{kind},{subkind},d,s,i\n",
        f"class,subclass,r_id,r_name,direction,generalises\n"
        f"{kind},{subkind},Is a,Is a,{direction},true\n",
    )

    if valid:
        read_classification_files(tmp_path)
    else:
        with pytest.raises(ClassificationDataError, match="cannot generalise: \\['Is a'\\]"):
            read_classification_files(tmp_path)


@pytest.mark.parametrize("subclass", ["unknown", ""])
def test_a_mapping_row_without_a_classification_is_rejected(tmp_path, subclass):
    _write(
        tmp_path,
        "Identity,x,d,s,i\n",
        f"class,subclass,r_id,r_name\nIdentity,x,Maps to,Maps to\nIdentity,{subclass},Is a,Is a\n",
    )

    with pytest.raises(ClassificationDataError, match="without a classification: \\['Is a'\\]"):
        read_classification_files(tmp_path)


@pytest.mark.parametrize(
    ("kind", "direction", "message"),
    [
        (HIERARCHY, "", "without an up or down direction: \\['Is a'\\]"),
        (HIERARCHY, "sideways", "without an up or down direction: \\['Is a'\\]"),
        (IDENTITY_KIND, "up", "not Hierarchy but have a direction: \\['Is a'\\]"),
    ],
)
def test_only_hierarchy_relationships_have_a_direction(tmp_path, kind, direction, message):
    _write(
        tmp_path,
        f"{kind},x,d,s,i\n",
        f"class,subclass,r_id,r_name,direction\n{kind},x,Is a,Is a,{direction}\n",
    )

    with pytest.raises(ClassificationDataError, match=message):
        read_classification_files(tmp_path)


def test_the_command_prints_a_classification_error_and_exits_without_a_traceback(monkeypatch):
    from typer.testing import CliRunner

    from omop_graph import cli

    def fail(*args, **kwargs):
        raise ClassificationDataError("predicate_mapping.csv is invalid")

    monkeypatch.setattr(cli, "relationship_classification", fail)
    monkeypatch.setattr(cli.OmopGraphConfig, "configure_logging", lambda *args, **kwargs: None)

    result = CliRunner().invoke(cli.app, ["relationship-classification"])

    assert result.exit_code == 1
    assert "predicate_mapping.csv is invalid" in result.output
    assert "Traceback" not in result.output
