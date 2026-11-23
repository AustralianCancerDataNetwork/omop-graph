"""Checks that the adapter honours the oaklib interfaces it claims.

Every method that raises ``NotImplementedError`` in a claimed oaklib interface must be
implemented by the adapter or registered in ``UNSUPPORTED``. Methods still to be done are
listed in ``PENDING`` and shrink as the phases land: a pending method that gets implemented
fails its strict xfail until it is removed from the list.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterator

import pytest
from oaklib.interfaces import BasicOntologyInterface, SearchInterface, TextAnnotatorInterface

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.unsupported import UNSUPPORTED

CLAIMED = (BasicOntologyInterface, SearchInterface, TextAnnotatorInterface)

PENDING = frozenset(
    {
        "BasicOntologyInterface._all_entailed_relationships",
        "BasicOntologyInterface._all_relationships",
        "BasicOntologyInterface.comments",
        "BasicOntologyInterface.definition",
        "BasicOntologyInterface.multilingual_labels",
        "BasicOntologyInterface.obsoletes",
        "BasicOntologyInterface.ontologies",
        "BasicOntologyInterface.ontology_metadata_map",
        "BasicOntologyInterface.ontology_versions",
        "BasicOntologyInterface.owl_types",
        "BasicOntologyInterface.relationships_metadata",
        "BasicOntologyInterface.simple_mappings_by_curie",
        "BasicOntologyInterface.subset_members",
        "BasicOntologyInterface.subsets",
        "BasicOntologyInterface.terms_categories",
    }
)


def _raises_not_implemented(function: Callable) -> bool:
    """True when the body of *function* is only a ``raise NotImplementedError``."""
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    except (OSError, TypeError, SyntaxError):
        return False
    if not isinstance(node, ast.FunctionDef):
        return False
    body = [
        statement
        for statement in node.body
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
    ]
    return (
        len(body) == 1
        and isinstance(body[0], ast.Raise)
        and "NotImplementedError" in ast.dump(body[0])
    )


def _claimed_methods() -> Iterator[tuple[type, str, Callable]]:
    for interface in CLAIMED:
        for name, member in vars(interface).items():
            if inspect.isfunction(member) and not name.startswith("__"):
                yield interface, name, member


def _unimplemented_cases() -> Iterator[pytest.ParameterSet]:
    for interface, name, member in _claimed_methods():
        if not _raises_not_implemented(member):
            continue
        key = f"{interface.__name__}.{name}"
        marks = [pytest.mark.xfail(strict=True, reason="pending")] if key in PENDING else []
        yield pytest.param(interface, name, id=key, marks=marks)


def _is_compatible(base: inspect.Signature, override: inspect.Signature) -> bool:
    """True when *override* accepts every call that *base* accepts."""
    base_params = list(base.parameters.values())[1:]
    override_params = list(override.parameters.values())[1:]
    if any(p.kind is p.VAR_POSITIONAL for p in override_params):
        return True
    for position, expected in enumerate(base_params):
        if position >= len(override_params):
            return False
        actual = override_params[position]
        if actual.name != expected.name:
            return False
        if expected.default is not inspect.Parameter.empty and actual.default is actual.empty:
            return False
    return all(
        p.default is not p.empty or p.kind is p.VAR_KEYWORD
        for p in override_params[len(base_params) :]
    )


def test_the_claimed_interfaces_are_listed_as_implemented(mock_cdm_kg: KnowledgeGraph) -> None:
    adapter = OMOPAlchemyImplementation(kg=mock_cdm_kg)

    assert set(CLAIMED) <= set(adapter.interfaces_implemented())
    assert all(isinstance(adapter, interface) for interface in CLAIMED)


@pytest.mark.parametrize(
    ("interface", "name", "base"),
    [pytest.param(i, n, m, id=f"{i.__name__}.{n}") for i, n, m in _claimed_methods()],
)
def test_overrides_accept_every_call_the_interface_accepts(
    interface: type, name: str, base: Callable
) -> None:
    override = getattr(OMOPAlchemyImplementation, name)

    assert override is base or _is_compatible(inspect.signature(base), inspect.signature(override))


@pytest.mark.parametrize(("interface", "name"), _unimplemented_cases())
def test_methods_that_raise_in_oaklib_are_implemented_or_registered(
    interface: type, name: str
) -> None:
    override = inspect.getattr_static(OMOPAlchemyImplementation, name)
    function = override.fget if isinstance(override, property) else override

    assert function is not vars(interface)[name]
    assert f"{interface.__name__}.{name}" in UNSUPPORTED or not _raises_not_implemented(function)
