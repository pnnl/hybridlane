# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math
from functools import reduce
from typing import Any, Hashable, Iterable, Optional, Sequence

import pennylane as qml
from pennylane.operation import CVOperation, Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from ..sa import ComputationalBasis
from .mixins import Spectral


def _can_replace(x, y):
    """
    Convenience function that returns true if x is close to y and if
    x does not require grad
    """
    return (
        not qml.math.is_abstract(x)
        and not qml.math.requires_grad(x)
        and qml.math.allclose(x, y)
    )


class TwoModeSum(CVOperation):
    r"""Two-mode summing gate :math:`SUM(\lambda)`

    This continuous-variable gate implements the unitary

    .. math::

        SUM(\lambda) = \exp[\frac{\lambda}{2}(a + a^\dagger)(b^\dagger - b)]

    where :math:`\lambda \in \mathbb{R}` is a real parameter. The action on the wavefunction is given by

    .. math::

        SUM(\lambda)\ket{x_a}\ket{x_b} = \ket{x_a}\ket{x_b + \lambda x_a}

    in the position basis (see Box III.6 of [1]_).

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 2

    def __init__(self, lambda_: TensorLike, wires: WiresLike, id: Optional[str] = None):
        shape = qml.math.shape(lambda_)
        if len(shape) > 1:
            raise ValueError(f"Expected a scalar value; got shape {shape}")

        super().__init__(lambda_, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

    def adjoint(self):
        lambda_ = self.parameters[0]
        return TwoModeSum(-lambda_, wires=self.wires)

    def pow(self, z: int | float):
        return [TwoModeSum(self.data[0] * z, self.wires)]

    def simplify(self):
        lambda_ = self.data[0]
        if _can_replace(lambda_, 0):
            return qml.Identity(self.wires)

        return TwoModeSum(lambda_, self.wires)


class ModeSwap(CVOperation):
    r"""Continuous-variable SWAP between two qumodes

    The unitary implementing this gate is

    .. math::

        SWAP = \exp[\frac{\pi}{2}(a^\dagger b - ab^\dagger)]

    (see Box III.4 of [1]_). This is a special case of the :py:class:`~pennylane.ops.cv.Beamsplitter` gate with :math:`SWAP = BS(\theta=\pi/2, \varphi=-\pi)`.

    .. note::

        Pennylane uses a different convention for the Beamsplitter gate compared to the CVDV paper. In particular,
        one must transform :math:`\theta \rightarrow \theta/2` and :math:`\varphi \rightarrow -(\varphi + \pi/2)` to be compatible
        with the original definition.

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 2

    def __init__(self, wires: WiresLike, id: Optional[str] = None):
        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(*params, wires, **hyperparameters):
        return [qml.Beamsplitter(math.pi / 2, -math.pi, wires)]

    def adjoint(self):
        return ModeSwap(self.wires)  # self-adjoint up to a global phase of -1

    def pow(self, z: int | float):
        if isinstance(z, float):
            raise NotImplementedError("Unknown formula for fractional powers")
        elif z < 0:
            raise NotImplementedError("Unknown formula for inverse")

        if z % 2 == 0:
            return [qml.Identity(self.wires)]
        else:
            return [ModeSwap(self.wires)]


class Fourier(CVOperation):
    r"""Continuous-variable Fourier gate

    This gate is a special case of the CV :py:class:`~qml.Rotation` gate with :math:`\theta = \pi/2`
    """

    num_wires = 1

    def __init__(self, wires: WiresLike, id: Optional[str] = None):
        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(
        *params, wires, **hyperparameters
    ) -> Sequence[CVOperation]:
        return [
            qml.Rotation(-math.pi / 2, wires)
        ]  # minus sign because pennylane gates have +i instead of -i

    def adjoint(self):
        return qml.Rotation(math.pi / 2, self.wires)


# ------------------------------------
#           CV Observables
# ------------------------------------


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


class QuadP(qml.QuadP, Spectral):
    natural_basis = ComputationalBasis.Position  # type: ignore

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[qml.operation.Operator]:
        return [qml.Rotation(-math.pi / 2, wires=wires)]  # rotate p -> x

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires | Iterable[Hashable] | Hashable | None = None,
        **hyperparameters: dict[str, Any],
    ) -> list[qml.operation.Operator]:
        return [qml.Rotation(math.pi / 2, wires=wires), QuadX(wires)]

    def __repr__(self):
        inner = ", ".join(map(str, self.wires))
        return f"p̂({inner})"


class QuadOperator(qml.QuadOperator, Spectral):
    r"""The generalized quadrature observable :math:`\hat{x}_\phi = \hat{x} \cos\phi + \hat{p} \sin\phi`.

    When used with the :func:`~pennylane.expval` function, the expectation
    value :math:`\braket{\hat{x_\phi}}` is returned. This corresponds to
    the mean displacement in the phase space along axis at angle :math:`\phi`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Observable order: 1st order in the quadrature operators
    * Heisenberg representation:

      .. math:: d = [0, \cos\phi, \sin\phi]

    Args:
        phi (float): axis in the phase space at which to calculate
            the generalized quadrature observable
        wires (Sequence[Any] or Any): the wire the operation acts on
        id (str or None): String representing the operation (optional)"""

    natural_basis = ComputationalBasis.Position  # type: ignore

    @staticmethod
    def compute_diagonalizing_gates(
        *params: TensorLike,
        wires: Wires | Iterable[Hashable] | Hashable,
        **hyperparams: dict[str, Any],
    ) -> list[qml.operation.Operator]:
        return [qml.Rotation(-params[0], wires)]

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires | Iterable[Hashable] | Hashable | None = None,
        **hyperparameters: dict[str, Any],
    ) -> list[qml.operation.Operator]:
        return [qml.Rotation(params[0], wires=wires), QuadX(wires)]


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
