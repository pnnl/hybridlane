# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math

import numpy as np
import pennylane as qml
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    pow_rotation,
)
from pennylane.operation import CVOperation
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike
from typing_extensions import override


from ..op_math.decompositions.qubit_conditioned_decompositions import to_native_qcond
from .heisenberg import rotation


# Re-export since it matches the convention of Y. Liu
class Displacement(CVOperation):
    r"""Phase space displacement gate :math:`D(\alpha)`

    .. math::
       D(\alpha) = \exp[\alpha \ad -\alpha^* a]

    where :math:`\alpha = ae^{i\phi}` (see Box IV.1 [1]_). The result of applying a displacement to the vacuum
    is a coherent state :math:`D(\alpha)\ket{0} = \ket{\alpha}`. It has the symplectic representation

    .. math::

        D^\dagger(\alpha)\hat{x}D(\alpha) = \hat{x} + \mathrm{Re}[\alpha] \\
        D^\dagger(\alpha)\hat{p}D(\alpha) = \hat{p} + \mathrm{Im}[\alpha]

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 2
    num_wires = 1
    ndim_params = (0, 0)
    grad_method = qml.Displacement.grad_method
    grad_recipe = qml.Displacement.grad_recipe

    resource_keys = set()

    def __init__(
        self, a: TensorLike, phi: TensorLike, wires: WiresLike, id: str | None = None
    ):
        super().__init__(a, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    @override
    @staticmethod
    def _heisenberg_rep(p):  # pyright: ignore[reportIncompatibleMethodOverride]
        a = p[0]
        c = np.cos(p[1])
        s = np.sin(p[1])
        return np.array(
            [
                [1, 0, 0],
                [a * c, 1, 0],
                [a * s, 0, 1],
            ]
        )

    @override
    def adjoint(self):
        a, phi = self.parameters
        return Displacement(-a, phi, wires=self.wires)

    @override
    def pow(self, z: int | float):
        return [Displacement(z * self.data[0], self.data[1], self.wires)]

    @override
    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * np.pi)

        if _can_replace(a, 0):
            return qml.Identity(wires=self.wires)

        return Displacement(a, phi, self.wires)

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "D", cache=cache
        )


@qml.register_resources({Displacement: 1})
def _pow_d(a, phi, wires, z, **_):
    Displacement(a * z, phi, wires)


qml.add_decomps("Adjoint(Displacement)", adjoint_rotation)
qml.add_decomps("Pow(Displacement)", _pow_d)
qml.add_decomps("qCond(Displacement)", to_native_qcond(1))


# Modify to use -i convention
class Rotation(CVOperation):
    r"""Phase-space rotation gate :math:`R(\theta)`

    It is given by the unitary

    .. math::

        R(\theta) = \exp[-i\theta \hat{n}]

    where :math:`\theta \in [0, 2\pi)` (Box IV.2 [1]_). It has the symplectic representation

    .. math::

        \begin{pmatrix} \hat{x}(\theta) \\ \hat{p}(\theta) \end{pmatrix} =
        \begin{pmatrix} \cos(\theta) & \sin(\theta) \\ -\sin(\theta) & \cos(\theta) \end{pmatrix}
        \begin{pmatrix} \hat{x} \\ \hat{p} \end{pmatrix}

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 1
    ndim_params = (0,)
    num_wires = 1
    grad_method = qml.Rotation.grad_method
    grad_recipe = qml.Rotation.grad_recipe

    resource_keys = set()

    def __init__(self, theta: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(theta, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    @override
    @staticmethod
    def _heisenberg_rep(p):  # pyright: ignore[reportIncompatibleMethodOverride]
        return rotation(p[0])

    @override
    def adjoint(self):
        return Rotation(-self.parameters[0], wires=self.wires)

    @override
    def pow(self, z: int | float):
        return [Rotation(z * self.data[0], self.wires)]

    @override
    def simplify(self):
        theta = self.data[0] % (2 * np.pi)
        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return Rotation(theta, self.wires)

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "R", cache=cache
        )


qml.add_decomps("Adjoint(Rotation)", adjoint_rotation)
qml.add_decomps("Pow(Rotation)", pow_rotation)
qml.add_decomps("qCond(Rotation)", to_native_qcond(1))


# Re-export since it matches paper convention
class Squeezing(CVOperation):
    r"""Phase space squeezing gate :math:`S(\zeta)`

    It is given by the unitary

    .. math::
        S(\zeta) = \exp\left[\frac{1}{2}(\zeta^* a^2 - \zeta(\ad)^2)\right].

    where :math:`\zeta = r e^{i\phi}` (Box IV.3 [1]_). It has the symplectic representation

    .. math::

        (\cos\theta \hat{x} + \sin\theta \hat{p}) \rightarrow e^{-r} (\cos\theta \hat{x} + \sin\theta \hat{p}) \\
        (-\sin\theta \hat{x} + \cos\theta \hat{p}) \rightarrow e^{+r} (-\sin\theta \hat{x} + \cos\theta \hat{p})

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 2
    num_wires = 1
    ndim_params = (0, 0)
    grad_method = qml.Squeezing.grad_method
    grad_recipe = qml.Squeezing.grad_recipe

    resource_keys = set()

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    @override
    @staticmethod
    def _heisenberg_rep(p):  # pyright: ignore[reportIncompatibleMethodOverride]
        R = rotation(p[1])
        return R @ np.diag([1, math.exp(-p[0]), math.exp(p[0])]) @ R.T

    @override
    def adjoint(self):
        r, phi = self.parameters
        return Squeezing(-r, phi, wires=self.wires)

    @override
    def pow(self, z: int | float):
        return [Squeezing(z * self.data[0], self.data[1], self.wires)]

    @override
    def simplify(self):
        r, phi = self.data[0], self.data[1] % (2 * np.pi)

        if _can_replace(r, 0):
            return qml.Identity(wires=self.wires)

        return Squeezing(r, phi, self.wires)

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "S", cache=cache
        )


@qml.register_resources({Squeezing: 1})
def _pow_s(r, phi, wires, z, **_):
    Squeezing(r * z, phi, wires)


qml.add_decomps("Adjoint(Squeezing)", adjoint_rotation)
qml.add_decomps("Pow(Squeezing)", _pow_s)
qml.add_decomps("qCond(Squeezing)", to_native_qcond(1))


# Modify to have -i convention
class Kerr(CVOperation):
    r"""Kerr gate :math:`K(\kappa)`

    It is given by the unitary

    .. math::

        K(\kappa) = \exp[-i \kappa \hat{n}^2].

    Note that this differs from the self-Kerr interaction (eq. A25 [1]_) by an overall phase-space
    rotation gate, since

    .. math::

        e^{-i\kappa \ad\ad aa} &= e^{-i\kappa \hat{n}(\hat{n} - 1)} \\
                               &= e^{-i\kappa\hat{n}^2}e^{-i\kappa\hat{n}} \\
                               &= K(\kappa)R(\kappa)

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
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

    @override
    def adjoint(self):
        return Kerr(-self.parameters[0], wires=self.wires)

    @override
    def pow(self, z: int | float):
        return [Kerr(z * self.data[0], self.wires)]

    @override
    def simplify(self):
        kappa = self.data[0]
        if _can_replace(kappa, 0):
            return qml.Identity(wires=self.wires)

        return self

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "K", cache=cache
        )


qml.add_decomps("Adjoint(Kerr)", adjoint_rotation)
qml.add_decomps("Pow(Kerr)", pow_rotation)


# Modify for -i convention
class CubicPhase(CVOperation):
    r"""Cubic phase shift gate :math:`C(r)`

    It is given by the unitary (Table IV.2 [1]_)

    .. math::

        C(r) = e^{-i r \hat{x}^3}

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
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

    @override
    def adjoint(self):
        return CubicPhase(-self.parameters[0], wires=self.wires)

    @override
    def pow(self, z: int | float):
        return [CubicPhase(z * self.data[0], self.wires)]

    @override
    def simplify(self):
        r = self.data[0]
        if _can_replace(r, 0):
            return qml.Identity(wires=self.wires)

        return self

    @override
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
