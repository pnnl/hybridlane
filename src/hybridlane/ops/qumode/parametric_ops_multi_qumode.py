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


# Change to match convention
class Beamsplitter(CVOperation):
    r"""Beamsplitter gate :math:`BS(\theta, \varphi)`

    .. math::

        BS(\theta,\varphi) = \exp\left[-i \frac{\theta}{2} (e^{i\varphi} \ad b + e^{-i\varphi}ab^\dagger)\right]
    """

    num_params = 2
    num_wires = 2
    ndim_params = (0, 0)
    grad_method = "A"
    grad_recipe = (_two_term_shift_rule, _two_term_shift_rule)

    resource_keys = set()

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(theta, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

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


@qml.register_resources({Beamsplitter: 1})
def _pow_bs(theta, phi, wires, z, **_):
    Beamsplitter(theta * z, phi, wires)


qml.add_decomps("Adjoint(Beamsplitter)", adjoint_rotation)
qml.add_decomps("Pow(Beamsplitter)", _pow_bs)


# Re-export flipping sign of r, equivalent to φ -> φ + π
class TwoModeSqueezing(CVOperation):
    r"""Phase space two-mode squeezing :math:`TMS(r, \varphi)`

    .. math::

        TMS(r, \varphi) = \exp\left[r (e^{i\phi} \ad b^\dagger - e^{-i\phi} ab\right].
    """

    num_params = 2
    num_wires = 2
    ndim_params = (0, 0)
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / math.sinh(shift)
    a = 1
    grad_recipe = (
        [[multiplier, a, shift], [-multiplier, a, -shift]],
        _two_term_shift_rule,
    )

    resource_keys = set()

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

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


@qml.register_resources({TwoModeSqueezing: 1})
def _pow_tms(r, phi, wires, z, **_):
    TwoModeSqueezing(r * z, phi, wires)


qml.add_decomps("Adjoint(TwoModeSqueezing)", adjoint_rotation)
qml.add_decomps("Pow(TwoModeSqueezing)", _pow_tms)


class TwoModeSum(CVOperation):
    r"""Two-mode summing gate :math:`SUM(\lambda)`

    This continuous-variable gate implements the unitary

    .. math::

        SUM(\lambda) = \exp[\frac{\lambda}{2}(a + \ad)(b^\dagger - b)]

    where :math:`\lambda \in \mathbb{R}` is a real parameter. The action on the wavefunction is given by

    .. math::

        SUM(\lambda)\ket{x_a}\ket{x_b} = \ket{x_a}\ket{x_b + \lambda x_a}

    in the position basis (see Box III.6 of [1]_).

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 1
    num_wires = 2
    ndim_params = (0,)
    grad_method = "F"

    resource_keys = set()

    def __init__(self, lambda_: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(lambda_, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

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


qml.add_decomps("Adjoint(TwoModeSum)", adjoint_rotation)
qml.add_decomps("Pow(TwoModeSum)", pow_rotation)


def _can_replace(x, y):
    return (
        not qml.math.is_abstract(x)
        and not qml.math.requires_grad(x)
        and qml.math.allclose(x, y)
    )
