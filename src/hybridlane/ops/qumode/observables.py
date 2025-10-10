# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math
from collections.abc import Iterable, Sequence
from functools import reduce
from typing import Any, Hashable

import numpy as np
import pennylane as qml
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    make_pow_decomp_with_period,
    pow_involutory,
    pow_rotation,
    self_adjoint,
)
from pennylane.operation import CVOperation, Operator
from pennylane.ops.cv import _rotation, _two_term_shift_rule
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from ...sa import ComputationalBasis
from ..mixins import Spectral
from .parametric_ops_single_qumode import Rotation


class QuadX(qml.QuadX, Spectral):
    natural_basis = ComputationalBasis.Position  # type: ignore

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[Operator]:
        return []

    def position_spectrum(self, *basis_states) -> Sequence[float]:
        return basis_states[0]

    def __repr__(self):
        inner = ", ".join(map(str, self.wires))
        return f"x̂({inner})"


X = QuadX
r"""Position operator :math:`\hat{x}`

.. seealso::

    This is an alias for :class:`~hybridlane.QuadX`
"""


class QuadP(qml.QuadP, Spectral):
    natural_basis = ComputationalBasis.Position  # type: ignore

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[qml.operation.Operator]:
        return [Rotation(math.pi / 2, wires=wires)]  # rotate p -> x

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires | Iterable[Hashable] | Hashable | None = None,
        **hyperparameters: dict[str, Any],
    ) -> list[qml.operation.Operator]:
        return [Rotation(-math.pi / 2, wires=wires), QuadX(wires)]

    def __repr__(self):
        inner = ", ".join(map(str, self.wires))
        return f"p̂({inner})"


P = QuadP
r"""Momentum operator :math:`\hat{p}`

.. seealso::

    This is an alias for :class:`~hybridlane.QuadP`
"""


class QuadOperator(qml.QuadOperator, Spectral):
    r"""The generalized quadrature observable :math:`\hat{x}_\phi = \hat{x} \cos\phi + \hat{p} \sin\phi`

    When used with the :func:`~hybridlane.expval` function, the expectation
    value :math:`\braket{\hat{x_\phi}}` is returned. This corresponds to
    the mean displacement in the phase space along axis at angle :math:`\phi`.
    """

    natural_basis = ComputationalBasis.Position  # type: ignore

    @staticmethod
    def compute_diagonalizing_gates(
        *params: TensorLike,
        wires: Wires | Iterable[Hashable] | Hashable,
        **hyperparams: dict[str, Any],
    ) -> list[qml.operation.Operator]:
        return [Rotation(params[0], wires)]

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires | Iterable[Hashable] | Hashable | None = None,
        **hyperparameters: dict[str, Any],
    ) -> list[qml.operation.Operator]:
        return [qml.Rotation(-params[0], wires=wires), QuadX(wires)]


class NumberOperator(qml.NumberOperator, Spectral):
    natural_basis = ComputationalBasis.Discrete  # type: ignore

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[Operator]:
        return []

    def fock_spectrum(self, *basis_states) -> Sequence[float]:
        return basis_states[0]

    def __repr__(self):
        inner = ", ".join(map(str, self.wires))
        return f"n̂({inner})"


N = NumberOperator
r"""Number operator :math:`\hat{n}`

.. seealso::

    This is an alias for :class:`~hybridlane.NumberOperator`
"""


class FockStateProjector(qml.FockStateProjector, Spectral):
    natural_basis = ComputationalBasis.Discrete  # type: ignore

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_diagonalizing_gates(
        *parameters: TensorLike, wires: WiresLike, **hyperparameters
    ) -> list[Operator]:
        return []

    def fock_spectrum(self, *basis_states) -> Sequence[float]:
        results = []
        for n, wire_states in zip(self.data, basis_states):
            results.append(wire_states == n)

        return reduce(lambda x, y: x & y, results)
