from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface._compliance import oaklib_interface_classes
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.unsupported import UNSUPPORTED, OMOPUnsupportedError


@pytest.mark.parametrize("key", UNSUPPORTED)
def test_every_registered_method_exists_on_its_oaklib_interface(key: str) -> None:
    interface_name, method = key.partition("(")[0].split(".")

    assert hasattr(oaklib_interface_classes()[interface_name], method)


@pytest.mark.parametrize(
    "key", [key for key in UNSUPPORTED if "edge_index" not in key and "(" not in key]
)
def test_unsupported_methods_raise_with_the_registry_reason(
    mock_cdm_kg: KnowledgeGraph, key: str
) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)
    method = getattr(adapter, key.split(".")[1])

    with pytest.raises(OMOPUnsupportedError, match=UNSUPPORTED[key][:20]) as raised:
        method("x")

    assert isinstance(raised.value, NotImplementedError)


@pytest.mark.parametrize("name", ["edge_index", "entailed_edge_index"])
def test_edge_indexes_are_unsupported(mock_cdm_kg: KnowledgeGraph, name: str) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    with pytest.raises(OMOPUnsupportedError, match=name):
        getattr(adapter, name)


def test_a_method_unsupported_only_for_some_arguments_raises_for_those(
    mock_cdm_kg: KnowledgeGraph,
) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    with pytest.raises(OMOPUnsupportedError, match="shortest=False"):
        adapter.paths(["OMOP:1"], ["OMOP:2"], shortest=False)


@pytest.mark.parametrize("name", ["edge_index", "entailed_edge_index"])
def test_an_unsupported_property_looks_absent_to_introspection(
    mock_cdm_kg: KnowledgeGraph, name: str
) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    assert not hasattr(adapter, name)
    assert name not in dict(inspect.getmembers(adapter))


def test_there_are_no_definitions_to_list(mock_cdm_kg: KnowledgeGraph) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    assert list(adapter.definitions(["OMOP:1"], include_metadata=True)) == []
    assert list(adapter.definitions(["OMOP:1"], include_metadata=True, include_missing=True)) == [
        ("OMOP:1", None, {})
    ]
    assert list(adapter.definitions(["OMOP:1"], include_missing=True)) == [("OMOP:1", None)]


def test_every_error_raised_names_a_registered_method() -> None:
    sources = pathlib.Path(__file__).parents[2].joinpath("src/omop_graph/oaklib_interface")
    raised = {
        node.args[0].value
        for path in sources.glob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"OMOPUnsupportedError", "OMOPUnsupportedPropertyError"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }

    assert raised <= set(UNSUPPORTED)
    assert set(UNSUPPORTED) <= raised
