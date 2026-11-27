"""Checks that the adapter honours the oaklib interfaces it claims.

Every method that raises ``NotImplementedError``, whatever the arguments, in a claimed oaklib
interface must be implemented by the adapter or registered in ``UNSUPPORTED``. An override
that falls back to ``super()`` of a method that raises is not an implementation.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fixtures.helpers import docs_macros

from omop_graph.graph.kg import KnowledgeGraph
from omop_graph.oaklib_interface._compliance import (
    claimed_interface_classes,
    unclaimed_interface_names,
)
from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation
from omop_graph.oaklib_interface.unsupported import NOT_CLAIMED, UNSUPPORTED

CLAIMED = tuple(claimed_interface_classes().values())

AUDITED_GUARDS = {
    "EmbeddingProviderInterface.entity_embeddings": "raises only when the adapter is not an OboGraphInterface or a SemanticSimilarityInterface",
    "SemanticSimilarityInterface.setwise_most_recent_common_ancestors": "raises only when the adapter is not an OboGraphInterface",
    "SemanticSimilarityInterface.multiset_most_recent_common_ancestors": "raises only when the adapter is not an OboGraphInterface",
}
"""Methods whose oaklib raise sits behind a check of the adapter's own type that it passes."""


def _raises_not_implemented(function: Callable) -> bool:
    """True when *function* raises a bare ``NotImplementedError`` anywhere, also for some arguments."""
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    except (OSError, TypeError, SyntaxError):
        return False
    return any(
        isinstance(statement, ast.Raise)
        and isinstance(statement.exc, (ast.Name, ast.Call))
        and "NotImplementedError" in ast.dump(statement.exc)
        for statement in ast.walk(node)
    )


def _falls_back_to_super(function: Callable, name: str) -> bool:
    """True when *function* calls ``super().<name>(...)``, which is the method it overrides."""
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    except (OSError, TypeError, SyntaxError):
        return False
    return any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == name
        and isinstance(call.func.value, ast.Call)
        and isinstance(call.func.value.func, ast.Name)
        and call.func.value.func.id == "super"
        for call in ast.walk(node)
    )


def _claimed_methods() -> Iterator[tuple[type, str, Callable]]:
    for interface in CLAIMED:
        for name, member in vars(interface).items():
            if inspect.isfunction(member) and not name.startswith("__"):
                yield interface, name, member


def _claimed_members() -> Iterator[tuple[type, str, Callable]]:
    """The methods of the claimed interfaces, and the getters of their properties."""
    for interface in CLAIMED:
        for name, member in vars(interface).items():
            function = member.fget if isinstance(member, property) else member
            if inspect.isfunction(function) and not name.startswith("__"):
                yield interface, name, function


def _unimplemented_cases() -> Iterator[Any]:
    for interface, name, member in _claimed_members():
        if not _raises_not_implemented(member):
            continue
        yield pytest.param(interface, name, id=f"{interface.__name__}.{name}")


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
    key = f"{interface.__name__}.{name}"
    if key in AUDITED_GUARDS:
        pytest.skip(AUDITED_GUARDS[key])
    override = inspect.getattr_static(OMOPAlchemyImplementation, name)
    function = override.fget if isinstance(override, property) else override
    registered = any(entry.partition("(")[0] == key for entry in UNSUPPORTED)

    assert registered or function is not vars(interface)[name]
    assert registered or not _raises_not_implemented(function)
    assert registered or not _falls_back_to_super(function, name)


def test_a_fallback_to_a_super_method_is_recognised() -> None:
    class Base:
        def embed(self, text):
            raise NotImplementedError

    class Falls(Base):
        def embed(self, text):
            return super().embed(text)

    class Implements(Base):
        def embed(self, text):
            return [text]

    assert _falls_back_to_super(Falls.embed, "embed")
    assert not _falls_back_to_super(Implements.embed, "embed")


def test_every_oaklib_interface_is_implemented_or_listed_with_a_reason() -> None:
    assert unclaimed_interface_names() == set(NOT_CLAIMED)


def test_every_unsupported_method_belongs_to_an_implemented_interface() -> None:
    claimed = claimed_interface_classes()

    for key in UNSUPPORTED:
        interface, _, method = key.partition("(")[0].partition(".")
        assert hasattr(claimed[interface], method), key


def test_the_compliance_tables_list_the_registries() -> None:
    markdown = docs_macros()["oaklib_compliance"]()

    assert all(f"`{key}`" in markdown for key in UNSUPPORTED)
    assert all(f"`{name}`" in markdown for name in NOT_CLAIMED)
