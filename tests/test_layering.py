"""Importing a layer of the package loads only that layer and the ones below it."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

BELOW = {
    "omop_graph.extensions.relationship_classification": ("omop_graph.extensions",),
    "omop_graph.graph.model.edges": ("omop_graph.extensions", "omop_graph.graph.model"),
    "omop_graph.graph.queries": ("omop_graph.extensions", "omop_graph.graph.model"),
    "omop_graph.graph.capabilities.hierarchy": (
        "omop_graph.extensions",
        "omop_graph.graph.model",
        "omop_graph.graph.queries",
        "omop_graph.graph.capabilities",
    ),
    "omop_graph.graph.kg": (
        "omop_graph.extensions",
        "omop_graph.graph.model",
        "omop_graph.graph.queries",
        "omop_graph.graph.capabilities",
        "omop_graph.graph.kg",
    ),
}
"""For each module, the packages and modules that importing it may load."""


@pytest.mark.parametrize("module", BELOW)
def test_a_layer_imports_only_the_layers_below_it(module: str) -> None:
    program = (
        "import sys, json\n"
        f"import {module}\n"
        "print(json.dumps(sorted(n for n in sys.modules if n.startswith('omop_graph'))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(result.stdout))
    allowed = ("omop_graph", *BELOW[module])

    assert {
        name for name in loaded if not any(name == a or name.startswith(a + ".") for a in allowed)
    } == set()
