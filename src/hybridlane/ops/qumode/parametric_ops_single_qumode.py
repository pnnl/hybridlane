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
    ndim_params = (0, 0)
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / shift
    a = 1
    grad_recipe = (
        [[multiplier, a, shift], [-multiplier, a, -shift]],
        _two_term_shift_rule,
    )

    resource_keys = set()

    def __init__(
        self, a: TensorLike, phi: TensorLike, wires: WiresLike, id: str | None = None
    ):
        super().__init__(a, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

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


@qml.register_resources({Displacement: 1})
def _pow_d(a, phi, wires, z, **_):
    Displacement(a * z, phi, wires)


qml.add_decomps("Adjoint(Displacement)", adjoint_rotation)
qml.add_decomps("Pow(Displacement)", _pow_d)


# Modify to use -i convention
class Rotation(CVOperation):
    r"""Phase space rotation gate :math:`R(\theta)`

    .. math::

        R(\theta) = \exp[-i\theta \hat{n}]
    """

    num_params = 1
    num_wires = 1
    ndim_params = (0,)
    grad_method = "A"
    grad_recipe = (_two_term_shift_rule,)

    resource_keys = set()

    def __init__(self, theta: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(theta, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

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


qml.add_decomps("Adjoint(Rotation)", adjoint_rotation)
qml.add_decomps("Pow(Rotation)", pow_rotation)


# Re-export since it matches paper convention
class Squeezing(CVOperation):
    r"""Phase space squeezing gate :math:`S(\zeta)`

    .. math::
        S(\zeta) = \exp\left[\frac{1}{2}(\zeta^* a^2 - \zeta(\ad)^2)\right].

    where :math:`\zeta = r e^{i\phi}`.
    """

    num_params = 2
    num_wires = 1
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


@qml.register_resources({Squeezing: 1})
def _pow_s(r, phi, wires, z, **_):
    Squeezing(r * z, phi, wires)


qml.add_decomps("Adjoint(Squeezing)", adjoint_rotation)
qml.add_decomps("Pow(Squeezing)", _pow_s)


# Modify to have -i convention
class Kerr(CVOperation):
    r"""Kerr gate :math:`K(\kappa)`

    .. math::

        K(\kappa) = \exp[-i \kappa \hat{n}^2].
    """

    num_params = 1
    num_wires = 1
    ndim_params = (0,)
    grad_method = "F"

    resource_keys = set()

    def __init__(self, kappa: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(kappa, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

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


qml.add_decomps("Adjoint(Kerr)", adjoint_rotation)
qml.add_decomps("Pow(Kerr)", pow_rotation)


# Modify for -i convention
class CubicPhase(CVOperation):
    r"""Cubic phase shift gate :math:`C(r)`

    .. math::

        C(r) = e^{-i r \hat{x}^3}.
    """

    num_params = 1
    num_wires = 1
    ndim_params = (0,)
    grad_method = "F"

    resource_keys = set()

    def __init__(self, r: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(r, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

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


qml.add_decomps("Adjoint(CubicPhase)", adjoint_rotation)
qml.add_decomps("Pow(CubicPhase)", pow_rotation)


def _can_replace(x, y):
    return (
        not qml.math.is_abstract(x)
        and not qml.math.requires_grad(x)
        and qml.math.allclose(x, y)
    )
