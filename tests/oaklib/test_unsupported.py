from __future__ import annotations

import pytest
from oaklib import interfaces

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.unsupported import UNSUPPORTED, OMOPUnsupportedError


@pytest.mark.parametrize("key", UNSUPPORTED)
def test_every_registered_method_exists_on_its_oaklib_interface(key: str) -> None:
    interface_name, method = key.split(".")

    assert hasattr(getattr(interfaces, interface_name), method)


@pytest.mark.parametrize("key", [key for key in UNSUPPORTED if "edge_index" not in key])
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
