# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Hashable, Sequence

from pennylane.wires import WireError, Wires


class ComputationalBasis(Enum):
    r"""Enum containing the different computational bases in CV-DV computing

    The discrete basis is used for the familiar computational basis of qubits, :math:`\{\ket{0}, \ket{1}\}`. It can also
    represent the result of measuring qudits :math:`\{\ket{0},\dots, \ket{d-1}\}` and Fock state measurements on qumodes
    :math:`\ket{n}, n \in \mathbb{N}`. The result of a discrete measurement is an ``int``.

    The position basis describes the continuous (qumode) basis :math:`\ket{x}`, where :math:`\ket{x}` is the non-normalizeable
    eigenket of the position operator :math:`\hat{x}\ket{x} = x\ket{x}`. This is an equivalent notion of a computational basis
    in CV computing, and it is implemented through homodyne detection. The result of a position basis measurement is a ``float``.

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

    def __init__(self, value, return_type: type):
        """
        Args:
            value: The internal value of the enum

            return_type: The type required to represent a basis vector
        """
        self._value_ = value
        self.return_type = return_type


# todo: maybe this should also be a basemodel
class BasisSchema:
    r"""Utility class for representing the computational basis that wires are measured in"""

    def __init__(self, wire_map: dict[Wires, ComputationalBasis]):
        self._wire_map: dict[Hashable, ComputationalBasis] = {}
        for wires, basis in wire_map.items():
            if not isinstance(basis, ComputationalBasis):
                raise ValueError("All bases must be a ComputationalBasis object")

            for wire in wires:
                self._wire_map[wire] = basis

    def get_basis(self, wire: Hashable) -> ComputationalBasis:
        r"""Gets the basis a particular wire is measured in"""
        return self._wire_map[wire]

    def get_type(self, wire: Hashable) -> type:
        r"""Gets the primitive data type for a wire"""
        return self._wire_map[wire].return_type

    @property
    def wires(self):
        return Wires.all_wires(self._wire_map.keys())

    @staticmethod
    def all_wires(schemas: Sequence["BasisSchema"]) -> "BasisSchema":
        return reduce(lambda x, y: x.union(y), schemas)

    @staticmethod
    def unique_wires(schemas: Sequence["BasisSchema"]) -> "BasisSchema":
        return reduce(lambda x, y: x.symmetric_difference(y), schemas)

    @staticmethod
    def common_wires(schemas: Sequence["BasisSchema"]) -> "BasisSchema":
        return reduce(lambda x, y: x.intersection(y), schemas)

    def intersection(self, other: "BasisSchema") -> "BasisSchema":
        common_wires = self.wires & other.wires
        for w in common_wires:
            if self.get_basis(w) != other.get_basis(w):
                raise WireError(f"Incompatible schemas on wire {w}")

        return self.for_wires(common_wires)

    def union(self, other: "BasisSchema") -> "BasisSchema":
        # Check for any conflicts
        for w in self.wires & other.wires:
            if self.get_basis(w) != other.get_basis(w):
                raise WireError(f"Incompatible schemas on wire {w}")

        return BasisSchema(
            {Wires(k): v for k, v in (self._wire_map | other._wire_map).items()}
        )

    def difference(self, other: "BasisSchema") -> "BasisSchema":
        wires = self.wires - other.wires
        return self.for_wires(wires)

    def symmetric_difference(self, other: "BasisSchema") -> "BasisSchema":
        wires = self.wires ^ other.wires
        return self.for_wires(self.wires & wires) + other.for_wires(other.wires & wires)

    def for_wires(self, wires: Wires) -> "BasisSchema":
        if unspecified_wires := wires - self.wires:
            raise WireError(f"Schema does not contain wires {unspecified_wires}")

        new_wiremap = {Wires(w): self.get_basis(w) for w in wires}
        return BasisSchema(new_wiremap)

    def __bool__(self):
        return bool(self.wires)

    def __and__(self, other: "BasisSchema"):
        return self.intersection(other)

    def __or__(self, other: "BasisSchema"):
        return self.union(other)

    def __xor__(self, other: "BasisSchema"):
        return self.symmetric_difference(other)

    def __add__(self, other: "BasisSchema"):
        return self.union(other)

    def __radd__(self, other: "BasisSchema"):
        return self.union(other)

    def __sub__(self, other: "BasisSchema"):
        return self.difference(other)

    def __rsub__(self, other: "BasisSchema"):
        return other.difference(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BasisSchema):
            return False

        return self._wire_map == other._wire_map


@dataclass
class StaticAnalysisResult:
    """Represents the result of static analysis on a quantum circuit."""

    qumodes: Wires
    """The inferred qumodes in the circuit"""

    qubits: Wires
    """The inferred qubits in the circuit"""

    schemas: list[BasisSchema | None]
    """The inferred schemas for each measurement process, in the same order as the circuit"""

    wire_order: Wires
    """The wire order for the original tape"""
