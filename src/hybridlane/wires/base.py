# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Base type definitions for type checking"""

from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import partial, reduce
from typing import Any

import pennylane as qp
from pennylane.wires import WireError, Wires, WiresLike


@dataclass(frozen=True)
class Qubit:
    """Type representing a qubit"""

    @property
    def supported_bases(self) -> tuple["ComputationalBasis", ...]:  # noqa: D102
        return (ComputationalBasis.Discrete,)


# Put here for the future; currently unused. Will require rethinking how
# to define wire type signatures in each operator
@dataclass(frozen=True)
class Qudit:
    """Type representing a qudit with specified dimension"""

    dim: int

    @property
    def supported_bases(self) -> tuple["ComputationalBasis", ...]:  # noqa: D102
        return (ComputationalBasis.Discrete,)


@dataclass(frozen=True)
class Qumode:
    """Type representing a qumode"""

    @property
    def supported_bases(self) -> tuple["ComputationalBasis", ...]:  # noqa: D102
        return (
            ComputationalBasis.Discrete,
            ComputationalBasis.Position,
            ComputationalBasis.Coherent,
        )


WireType = Qubit | Qudit | Qumode


class ComputationalBasis(Enum):
    r"""Enum containing the different computational bases in CV-DV computing

    The discrete basis is used for the familiar computational basis of qubits,
    :math:`\{\ket{0}, \ket{1}\}`. It can also represent the result of measuring qudits
    :math:`\{\ket{0},\dots, \ket{d-1}\}` and Fock state measurements on qumodes
    :math:`\ket{n}, n \in \mathbb{N}`. The result of a discrete measurement is an ``int``.

    The position basis describes the continuous (qumode) basis :math:`\ket{x}`, where
    :math:`\ket{x}` is the non-normalizeable eigenket of the position operator
    :math:`\hat{x}\ket{x} = x\ket{x}`. This is an equivalent notion of a computational basis
    in CV computing, and it is implemented through homodyne detection. The result of a position
    basis measurement is a ``float``.

    Finally, the coherent basis captures heterodyne detection, which measures the Husimi Q-function

    .. math::

        p(\alpha) = \frac{1}{\pi}Tr[\rho \ket{\alpha}\bra{\alpha}].

    The resulting type is a ``complex``.
    """

    Discrete = (1, int)
    r"""Countable, discrete energy eigenstate basis :math:`\ket{0}, \ket{1}, \dots`"""

    Position = (2, float)
    r"""Continuous position space along :math:`\ket{x}`"""

    Coherent = (3, complex)
    r"""Basis of coherent states :math:`\ket{\alpha}`"""

    def __init__(self, value, return_type: type):  # noqa: D107
        self._value_ = value
        self.return_type = return_type


class BasisMap(Mapping[Any, ComputationalBasis]):
    r"""Utility class for representing the computational basis that wires are measured in"""

    def __init__(self, wire_map: dict[WiresLike, ComputationalBasis]):  # noqa: D107
        self._wire_map: dict[Wires, ComputationalBasis] = {}
        for wires, basis in wire_map.items():
            for wire in Wires(wires):
                self._wire_map[wire] = basis

    def get_basis(self, wire: WiresLike) -> ComputationalBasis:
        r"""Gets the basis a particular wire is measured in"""
        return self._wire_map[wire]  # ty:ignore[invalid-argument-type]

    def get_type(self, wire: WiresLike) -> type:
        r"""Gets the primitive data type for a wire"""
        return self._wire_map[wire].return_type  # ty:ignore[invalid-argument-type]

    @property
    def wires(self) -> Wires:
        r"""Returns the wires in this schema"""
        return Wires.all_wires(self._wire_map.keys())

    @staticmethod
    def all_wires(maps: Sequence["BasisMap"]) -> "BasisMap":
        r"""Unions multiple basis maps together"""
        return reduce(lambda x, y: x.union(y), maps)

    @staticmethod
    def unique_wires(schemas: Sequence["BasisMap"]) -> "BasisMap":
        r"""Returns a map containing only the wires unique to each map"""
        return reduce(lambda x, y: x.symmetric_difference(y), schemas)

    @staticmethod
    def common_wires(schemas: Sequence["BasisMap"]) -> "BasisMap":
        r"""Returns a map containing only the wires common to all maps"""
        return reduce(lambda x, y: x.intersection(y), schemas)

    def intersection(self, other: "BasisMap") -> "BasisMap":
        r"""Returns a map containing only the wires common to both maps"""
        common_wires = self.wires & other.wires
        for w in common_wires:
            if self.get_basis(w) != other.get_basis(w):
                raise WireError(f"Incompatible schemas on wire {w}")

        return self.for_wires(common_wires)

    def union(self, other: "BasisMap") -> "BasisMap":
        r"""Returns a map containing all wires from both maps"""
        # Check for any conflicts
        for w in self.wires & other.wires:
            if self.get_basis(w) != other.get_basis(w):
                raise WireError(f"Incompatible schemas on wire {w}")

        return BasisMap(self._wire_map | other._wire_map)  # ty:ignore[invalid-argument-type]

    def difference(self, other: "BasisMap") -> "BasisMap":
        r"""Returns a map containing only the wires in this map that are not in the other map"""
        wires = self.wires - other.wires
        return self.for_wires(wires)

    def symmetric_difference(self, other: "BasisMap") -> "BasisMap":
        r"""Returns a map containing only the wires that are in either map, but not both"""
        wires = self.wires ^ other.wires
        return self.for_wires(self.wires & wires) + other.for_wires(other.wires & wires)

    def for_wires(self, wires: Wires) -> "BasisMap":
        r"""Returns a map containing only the specified wires"""
        if unspecified_wires := wires - self.wires:
            raise WireError(f"Schema does not contain wires {unspecified_wires}")

        new_wiremap = {w: self.get_basis(w) for w in wires}
        return BasisMap(new_wiremap)

    def __bool__(self):
        return bool(self.wires)

    def __and__(self, other: "BasisMap"):
        return self.intersection(other)

    def __or__(self, other: "BasisMap"):
        return self.union(other)

    def __xor__(self, other: "BasisMap"):
        return self.symmetric_difference(other)

    def __add__(self, other: "BasisMap"):
        return self.union(other)

    def __radd__(self, other: "BasisMap"):
        return self.union(other)

    def __sub__(self, other: "BasisMap"):
        return self.difference(other)

    def __rsub__(self, other: "BasisMap"):
        return other.difference(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BasisMap):
            return False

        return self._wire_map == other._wire_map

    def __getitem__(self, wire: WiresLike) -> ComputationalBasis:
        return self._wire_map[wire]  # ty:ignore[invalid-argument-type]

    def __iter__(self):
        return iter(self._wire_map)

    def __len__(self):
        return len(self._wire_map)

    def __repr__(self) -> str:
        if not self._wire_map:
            return "BasisMap({})"

        # Group wires by basis for more compact representation
        basis_to_wires: dict[ComputationalBasis, list] = {}
        for wire, basis in sorted(self._wire_map.items(), key=lambda x: (str(type(x[0])), x[0])):
            basis_to_wires.setdefault(basis, []).append(wire)

        # Build the wire_map dict representation
        items = []
        for basis, wires in basis_to_wires.items():
            if len(wires) == 1:
                items.append(f"{wires[0]!r}: ComputationalBasis.{basis.name}")
            else:
                # Use repr for the list to properly quote strings
                items.append(f"{wires!r}: ComputationalBasis.{basis.name}")

        wire_map_str = "{" + ", ".join(items) + "}"
        return f"BasisMap({wire_map_str})"

    def __str__(self) -> str:
        if not self._wire_map:
            return "BasisMap(empty)"

        # Group wires by basis type for readable output
        basis_to_wires: dict[ComputationalBasis, list] = {}
        for wire, basis in sorted(self._wire_map.items(), key=lambda x: (str(type(x[0])), x[0])):
            basis_to_wires.setdefault(basis, []).append(wire)

        parts = []
        for basis, wires in basis_to_wires.items():
            # Just convert to string - strings display without quotes, ints as-is
            wire_list = ", ".join(str(w) for w in wires)
            parts.append(f"{basis.name}: [{wire_list}]")

        return "BasisMap(" + ", ".join(parts) + ")"


@dataclass
class TypeCheckResult:
    """Represents the result of static analysis on a quantum circuit."""

    wire_types: OrderedDict[WiresLike, WireType]
    """The inferred type of each wire"""

    basis_maps: list[BasisMap | None]
    """The inferred schemas for each measurement process, in the same order as the circuit"""

    @property
    def qubits(self) -> Wires:
        r"""All qubits in the tape or operation"""
        return Wires([w for w, t in self.wire_types.items() if t == Qubit()])

    @property
    def qumodes(self) -> Wires:
        r"""All qumodes in the tape or operation"""
        return Wires([w for w, t in self.wire_types.items() if t == Qumode()])

    @property
    def wire_order(self) -> Wires:
        r"""The wires in the order they were first seen in the tape or operation"""
        return Wires(self.wire_types)


# experimental
class TypedWires(Sequence["TypedWire"], Hashable):
    """A register of wires with an associated type"""

    def __init__(self, wires: WiresLike, wire_type: WireType):  # noqa: D107
        self.wires = Wires(wires)
        self.wire_type = wire_type

    def __getitem__(self, key):
        return TypedWire(self.wires[key], self.wire_type)

    def __len__(self):
        return len(self.wires)

    def __repr__(self) -> str:
        return f"TypedWires(wires={self.wires!r}, wire_type={self.wire_type!r})"

    def __str__(self) -> str:
        return f"{self.wires} ({self.wire_type})"

    def __hash__(self):
        return hash((self.wires, self.wire_type))


# experimental
class TypedWire(Hashable):
    """A single wire with an associated type"""

    def __init__(self, wire: WiresLike, wire_type: WireType):  # noqa: D107
        wires = Wires(wire)
        if len(wires) != 1:
            raise ValueError("TypedWire must be initialized with a single wire")

        self.wire = wires[0]
        self.wire_type = wire_type

    def __repr__(self) -> str:
        return f"TypedWire(wire={self.wire!r}, wire_type={self.wire_type!r})"

    def __str__(self) -> str:
        return str(self.wire)

    def __hash__(self):
        return hash((self.wire, self.wire_type))


# experimental
def typed_registers(register_dict: dict, type: WireType) -> dict:
    r"""Creates a register of wires that have a particular, statically-declared type

    This function is analogous to :func:`pennylane.registers`, but the resulting wires are wrapped
    with a specific type, allowing you to declare a set of registers that have a particular type.

    This could be useful when defining a circuit that has uninformative operations (like Identity),
    and this may be used in the future for polymorphic operations.

    Args:
        register_dict: A dictionary of the format expected by ``qp.registers``

        type: The type of the wires in the register
    """
    possibly_nested_dict = qp.registers(register_dict)

    def remap_dict(d):
        if isinstance(d, Mapping):
            return {k: remap_dict(v) for k, v in d.items()}

        assert isinstance(d, Wires)
        return TypedWires(d, type)

    new_dict = {}
    for k, v in possibly_nested_dict.items():
        new_dict[k] = remap_dict(v)

    return new_dict


qubits = partial(typed_registers, type=Qubit())
"""Create a dictionary of qubit registers.

This has the same usage as :func:`pennylane.registers`, but the resulting wires are wrapped
in :class:`TypedWires` with type :class:`Qubit`.

**Example**

>>> reg = hl.qubits({"alice": 2})
{'alice': TypedWires(wires=Wires([0, 1]), wire_type=Qubit())}
"""

qumodes = partial(typed_registers, type=Qumode())
"""Create a dictionary of qumode registers.

This has the same usage as :func:`pennylane.registers`, but the resulting wires are wrapped
in :class:`TypedWires` with type :class:`Qumode`.

**Example**

>>> reg = hl.qumodes({"bob": 2})
{'bob': TypedWires(wires=Wires([0, 1]), wire_type=Qumode())}
"""
