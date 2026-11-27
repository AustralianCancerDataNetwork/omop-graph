"""Which oaklib interfaces the adapter implements, and which methods it cannot support."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass

import oaklib.interfaces as oaklib_interfaces
from oaklib.interfaces import OntologyInterface

from omop_graph.oaklib_interface.unsupported import UNSUPPORTED


@dataclass(frozen=True)
class ClaimedInterface:
    """
    An oaklib interface the adapter implements.

    Parameters
    ----------
    name : str
        The class name of the interface.
    methods : int
        The methods the interface defines.
    unsupported : int
        How many of them the adapter does not support.
    """

    name: str
    methods: int
    unsupported: int


def oaklib_interface_classes() -> dict[str, type[OntologyInterface]]:
    """Every interface oaklib defines below ``OntologyInterface``, keyed by class name."""
    found: dict[str, type[OntologyInterface]] = {}
    for module_info in pkgutil.iter_modules(oaklib_interfaces.__path__):
        module = importlib.import_module(f"{oaklib_interfaces.__name__}.{module_info.name}")
        for name, member in vars(module).items():
            if (
                inspect.isclass(member)
                and issubclass(member, OntologyInterface)
                and member is not OntologyInterface
                and member.__module__ == module.__name__
            ):
                found[name] = member
    return dict(sorted(found.items()))


def claimed_interface_classes() -> dict[str, type[OntologyInterface]]:
    """The oaklib interfaces the adapter implements, keyed by class name."""
    from omop_graph.oaklib_interface.omop_implementation import OMOPAlchemyImplementation

    return {
        name: interface
        for name, interface in oaklib_interface_classes().items()
        if issubclass(OMOPAlchemyImplementation, interface)
    }


def claimed_interfaces() -> list[ClaimedInterface]:
    """The claimed interfaces with their method counts."""
    return [
        ClaimedInterface(
            name=name,
            methods=sum(
                inspect.isfunction(member) and not attribute.startswith("__")
                for attribute, member in vars(interface).items()
            ),
            unsupported=sum(key.startswith(f"{name}.") for key in UNSUPPORTED),
        )
        for name, interface in claimed_interface_classes().items()
    ]


def unclaimed_interface_names() -> set[str]:
    """The oaklib interfaces the adapter does not implement."""
    return set(oaklib_interface_classes()) - set(claimed_interface_classes())
