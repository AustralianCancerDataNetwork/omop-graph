import logging

import pandas as pd
import pytest

from omop_graph.cli import _filter_unknown_relationship_ids


def test_unknown_relationship_ids_are_warned_about_and_dropped(caplog):
    mapping = pd.DataFrame(
        {
            "relationship_id": ["maps to", "unknown relationship"],
            "predicate_kind": ["Association", "Association"],
            "predicate_subkind": ["mapping", "mapping"],
        }
    )

    with caplog.at_level(logging.WARNING, logger="omop_graph.cli"):
        filtered = _filter_unknown_relationship_ids(mapping, {"maps to"})

    assert filtered["relationship_id"].tolist() == ["maps to"]
    assert "unknown relationship" in caplog.text


def test_empty_relationship_vocabulary_fails_clearly():
    mapping = pd.DataFrame({"relationship_id": ["maps to"]})

    with pytest.raises(RuntimeError, match="relationship vocabulary is empty"):
        _filter_unknown_relationship_ids(mapping, set())


def test_filter_fails_when_no_relationship_mapping_survives():
    mapping = pd.DataFrame({"relationship_id": ["unknown relationship"]})

    with pytest.raises(RuntimeError, match="None of the relationship mappings match"):
        _filter_unknown_relationship_ids(mapping, {"maps to"})
