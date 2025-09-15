# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
#
# Some gates are copied from Pennylane, which is subject to
# the Apache 2.0 license. They have been modified to match
# the definitions in the paper arXiv:2407.10381.
import math
from functools import reduce
from typing import Any, Hashable, Iterable, Optional, Sequence

import numpy as np
import pennylane as qml
from pennylane.operation import CVOperation, Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike
from pennylane.ops.cv import _rotation, _two_term_shift_rule

from ..sa import ComputationalBasis
from .mixins import Spectral


# Todo: Check the grad method/param shift rules/heisenberg rep of each operator


# Re-export since it matches the convention of Y. Liu
class Displacement(CVOperation):
    r"""Phase space displacement gate :math:`D(\alpha)`

    .. math::
       D(\alpha) = \exp[\alpha \ad -\alpha^* a]

    where :math:`\alpha = ae^{i\phi}`. The result of applying a displacement to the vacuum
    is a coherent state :math:`D(\alpha)\ket{0} = \ket{\alpha}`.
    """

    num_params = 2
    num_wires = 1
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / shift
    a = 1
    grad_recipe = (
        [[multiplier, a, shift], [-multiplier, a, -shift]],
        _two_term_shift_rule,
    )

    def __init__(
        self, a: TensorLike, phi: TensorLike, wires: WiresLike, id: Optional[str] = None
    ):
        super().__init__(a, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        c = math.cos(p[1])
        s = math.sin(p[1])
        scale = 2  # sqrt(2 * hbar)
        return np.array([[1, 0, 0], [scale * c * p[0], 1, 0], [scale * s * p[0], 0, 1]])

    def adjoint(self):
        a, phi = self.parameters
        new_phi = (phi + math.pi) % (2 * math.pi)
        return Displacement(a, new_phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "D", cache=cache
        )


# Modify to use -i convention
class Rotation(CVOperation):
    r"""Phase space rotation gate :math:`R(\theta)`

    .. math::

        R(\theta) = \exp[-i\theta \hat{n}]
    """

    num_params = 1
    num_wires = 1
    grad_method = "A"
    grad_recipe = (_two_term_shift_rule,)

    def __init__(self, theta: TensorLike, wires: WiresLike, id: Optional[str] = None):
        super().__init__(theta, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        return _rotation(-p[0])

    def adjoint(self):
        return Rotation(-self.parameters[0], wires=self.wires)

    def simplify(self):
        theta = self.data[0]
        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "R", cache=cache
        )


# Re-export since it matches paper convention
class Squeezing(CVOperation):
    r"""Phase space squeezing gate :math:`S(\zeta)`

    .. math::
        S(\zeta) = \exp\left[\frac{1}{2}(\zeta^* a^2 - \zeta(\ad)^2)\right].

    where :math:`\zeta = r e^{i\phi}`.
    """

    num_params = 2
    num_wires = 1
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / math.sinh(shift)
    a = 1
    grad_recipe = (
        [[multiplier, a, shift], [-multiplier, a, -shift]],
        _two_term_shift_rule,
    )

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1] / 2)
        return R @ np.diag([1, math.exp(-p[0]), math.exp(p[0])]) @ R.T

    def adjoint(self):
        r, phi = self.parameters
        new_phi = (phi + np.pi) % (2 * np.pi)
        return Squeezing(r, new_phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "S", cache=cache
        )


# Modify to have -i convention
class Kerr(CVOperation):
    r"""Kerr gate :math:`K(\kappa)`

    .. math::

        K(\kappa) = \exp[-i \kappa \hat{n}^2].
    """

    num_params = 1
    num_wires = 1
    grad_method = "F"

    def __init__(self, kappa: TensorLike, wires: WiresLike, id: Optional[str] = None):
        super().__init__(kappa, wires=wires, id=id)

    def adjoint(self):
        return Kerr(-self.parameters[0], wires=self.wires)

    def simplify(self):
        kappa = self.data[0]
        if _can_replace(kappa, 0):
            return qml.Identity(wires=self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "K", cache=cache
        )


# Modify for -i convention
class CubicPhase(CVOperation):
    r"""Cubic phase shift gate :math:`C(r)`

    .. math::

        C(r) = e^{-i r \hat{x}^3}.
    """

    num_params = 1
    num_wires = 1
    grad_method = "F"

    def __init__(self, r: TensorLike, wires: WiresLike, id: Optional[str] = None):
        super().__init__(r, wires=wires, id=id)

    def adjoint(self):
        return CubicPhase(-self.parameters[0], wires=self.wires)

    def simplify(self):
        r = self.data[0]
        if _can_replace(r, 0):
            return qml.Identity(wires=self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "C", cache=cache
        )


class Fourier(CVOperation):
    r"""Continuous-variable Fourier gate :math:`F`

    This gate is a special case of the CV :py:class:`~.Rotation` gate with :math:`\theta = \pi/2`
    """

    num_params = 0
    num_wires = 1

    def __init__(self, wires: WiresLike, id: Optional[str] = None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_decomposition(
        *params, wires, **hyperparameters
    ) -> Sequence[CVOperation]:
        return [Rotation(math.pi / 2, wires)]

    def adjoint(self):
        return Rotation(-math.pi / 2, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "F", cache=cache
        )


# Change to match convention
class Beamsplitter(CVOperation):
    r"""Beamsplitter gate :math:`BS(\theta, \varphi)`

    .. math::

        BS(\theta,\varphi) = \exp\left[-i \frac{\theta}{2} (e^{i\varphi} \ad b + e^{-i\varphi}ab^\dagger)\right]
    """

    num_params = 2
    num_wires = 2
    grad_method = "A"
    grad_recipe = (_two_term_shift_rule, _two_term_shift_rule)

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: Optional[str] = None,
    ):
        super().__init__(theta, phi, wires=wires, id=id)

    # For the beamsplitter, both parameters are rotation-like
    # Todo: Redo this with new convention
    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1], bare=True)
        c = math.cos(p[0])
        s = math.sin(p[0])
        U = c * np.eye(5)
        U[0, 0] = 1
        U[1:3, 3:5] = -s * R.T
        U[3:5, 1:3] = s * R
        return U

    def adjoint(self):
        theta, phi = self.parameters
        return Beamsplitter(-theta, phi, wires=self.wires)

    def simplify(self):
        theta, phi = self.data
        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "BS", cache=cache
        )


# Re-export flipping sign of r, equivalent to φ -> φ + π
class TwoModeSqueezing(CVOperation):
    r"""Phase space two-mode squeezing :math:`TMS(r, \varphi)`

    .. math::

        TMS(r, \varphi) = \exp\left[r (e^{i\phi} \ad b^\dagger - e^{-i\phi} ab\right].
    """

    num_params = 2
    num_wires = 2
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / math.sinh(shift)
    a = 1
    grad_recipe = (
        [[multiplier, a, shift], [-multiplier, a, -shift]],
        _two_term_shift_rule,
    )

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1] + np.pi, bare=True)

        S = math.sinh(p[0]) * np.diag([1, -1])
        U = math.cosh(p[0]) * np.identity(5)

        U[0, 0] = 1
        U[1:3, 3:5] = S @ R.T
        U[3:5, 1:3] = S @ R.T
        return U

    def adjoint(self):
        r, phi = self.parameters
        new_phi = (phi + np.pi) % (2 * np.pi)
        return TwoModeSqueezing(r, new_phi, wires=self.wires)

    def simplify(self):
        r = self.data[0]
        if _can_replace(r, 0):
            return qml.Identity(self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "TMS", cache=cache
        )


class TwoModeSum(CVOperation):
    r"""Two-mode summing gate :math:`SUM(\lambda)`

    This continuous-variable gate implements the unitary

    .. math::

        SUM(\lambda) = \exp[\frac{\lambda}{2}(a + \ad)(b^\dagger - b)]

    where :math:`\lambda \in \mathbb{R}` is a real parameter. The action on the wavefunction is given by

    .. math::

        SUM(\lambda)\ket{x_a}\ket{x_b} = \ket{x_a}\ket{x_b + \lambda x_a}

    in the position basis (see Box III.6 of [1]_).

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 1
    num_wires = 2

    def __init__(self, lambda_: TensorLike, wires: WiresLike, id: Optional[str] = None):
        super().__init__(lambda_, wires=wires, id=id)

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

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "SUM", cache=cache
        )


class ModeSwap(CVOperation):
    r"""Continuous-variable SWAP between two qumodes

    The unitary implementing this gate is

    .. math::

        SWAP = \exp[\frac{\pi}{2}(\ad b - ab^\dagger)]

    (see Box III.4 of [1]_). This is a special case of the :py:class:`~hybridlane.ops.cv.Beamsplitter` gate with :math:`SWAP = BS(\theta=\pi, \varphi=\pi / 2)`.

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 0
    num_wires = 2

    def __init__(self, wires: WiresLike, id: Optional[str] = None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def compute_decomposition(*params, wires, **hyperparameters):
        return [Beamsplitter(math.pi, math.pi / 2, wires)]

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
