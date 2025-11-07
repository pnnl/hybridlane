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
from pennylane.wires import Wires, WiresLike
from typing_extensions import override

from ..op_math.decompositions.qubit_conditioned_decompositions import to_native_qcond
from .heisenberg import is_symplectic, to_phase_space


# Change to match convention. See `from_pennylane` transform for more details
class Beamsplitter(CVOperation):
    r"""Beamsplitter gate :math:`BS(\theta, \varphi)`

    It is given by the unitary (Box IV.4 [1]_)

    .. math::

        BS(\theta,\varphi) = \exp\left[-i \frac{\theta}{2} (e^{i\varphi} \ad b + e^{-i\varphi}ab^\dagger)\right]

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 2
    num_wires = 2
    ndim_params = (0, 0)
    grad_method = qml.Beamsplitter.grad_method
    grad_recipe = qml.Beamsplitter.grad_recipe

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

    @override
    @staticmethod
    def _heisenberg_rep(p):
        # Comes from eqs. 170, 171
        c = np.cos(p[0] / 2)
        s = np.sin(p[0] / 2)
        pp = np.exp(1j * p[1])
        pm = np.conj(pp)

        S = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, c, 0, -1j * s * pp, 0],
                [0, 0, c, 0, 1j * s * pm],
                [0, -1j * s * pm, 0, c, 0],
                [0, 0, 1j * s * pp, 0, c],
            ]
        )
        return to_phase_space(S)

    @override
    def adjoint(self):
        theta, phi = self.parameters
        return Beamsplitter(-theta, phi, wires=self.wires)

    @override
    def pow(self, z: int | float):
        return [Beamsplitter(z * self.data[0], self.data[1], self.wires)]

    @override
    def simplify(self):
        theta, phi = self.data[0] % (4 * np.pi), self.data[1] % (2 * np.pi)
        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        return Beamsplitter(theta, phi, self.wires)

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "BS", cache=cache
        )


@qml.register_resources({Beamsplitter: 1})
def _pow_bs(theta, phi, wires, z, **_):
    Beamsplitter(theta * z, phi, wires)


qml.add_decomps("Adjoint(Beamsplitter)", adjoint_rotation)
qml.add_decomps("Pow(Beamsplitter)", _pow_bs)
qml.add_decomps("qCond(Beamsplitter)", to_native_qcond(1))


# Re-export flipping sign of r, equivalent to φ -> φ + π
class TwoModeSqueezing(CVOperation):
    r"""Phase space two-mode squeezing :math:`TMS(r, \varphi)`

    It is given by the unitary (Box IV.5 [1]_)

    .. math::

        TMS(r, \varphi) = \exp\left[r (e^{i\phi} \ad b^\dagger - e^{-i\phi} ab\right]

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 2
    num_wires = 2
    ndim_params = (0, 0)
    grad_method = qml.TwoModeSqueezing.grad_method
    grad_recipe = qml.TwoModeSqueezing.grad_recipe

    resource_keys = set()

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    @override
    @staticmethod
    def _heisenberg_rep(p):
        # See eq. 182, given in fock space
        S = math.cosh(p[0]) * np.identity(5, dtype=complex)
        S[0, 0] = 1
        S[(1, 2, 3, 4), (4, 3, 2, 1)] = math.sinh(p[0]) * np.exp(
            1j * p[1] * np.array([1, -1, 1, -1])
        )

        return to_phase_space(S)

    @override
    def adjoint(self):
        r, phi = self.parameters
        return TwoModeSqueezing(-r, phi, wires=self.wires)

    @override
    def pow(self, z: int | float):
        return [TwoModeSqueezing(z * self.data[0], self.data[1], self.wires)]

    @override
    def simplify(self):
        r, phi = self.data[0], self.data[1] % (2 * np.pi)
        if _can_replace(r, 0):
            return qml.Identity(self.wires)

        return TwoModeSqueezing(r, phi, self.wires)

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "TMS", cache=cache
        )


@qml.register_resources({TwoModeSqueezing: 1})
def _pow_tms(r, phi, wires, z, **_):
    TwoModeSqueezing(r * z, phi, wires)


qml.add_decomps("Adjoint(TwoModeSqueezing)", adjoint_rotation)
qml.add_decomps("Pow(TwoModeSqueezing)", _pow_tms)
qml.add_decomps("qCond(TwoModeSqueezing)", to_native_qcond(1))


class TwoModeSum(CVOperation):
    r"""Two-mode summing gate :math:`SUM(\lambda)`

    This continuous-variable gate implements the unitary

    .. math::

        SUM(\lambda) = \exp[\frac{\lambda}{2}(a + \ad)(b^\dagger - b)]

    where :math:`\lambda \in \mathbb{R}` is a real parameter. Note that this is in Wigner units.

    The action on the wavefunction is given by

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

    @override
    def adjoint(self):
        lambda_ = self.parameters[0]
        return TwoModeSum(-lambda_, wires=self.wires)

    @override
    def pow(self, z: int | float):
        return [TwoModeSum(self.data[0] * z, self.wires)]

    @override
    def simplify(self):
        lambda_ = self.data[0]
        if _can_replace(lambda_, 0):
            return qml.Identity(self.wires)

        return TwoModeSum(lambda_, self.wires)

    @override
    @staticmethod
    def _heisenberg_rep(p):
        # Defined in fock space, eq. B3
        l = p[0]
        S = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, -l / 2, +l / 2],
                [0, 0, 1, l / 2, -l / 2],
                [0, l / 2, l / 2, 1, 0],
                [0, l / 2, l / 2, 0, 1],
            ]
        )

        return to_phase_space(S)

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "SUM", cache=cache
        )


qml.add_decomps("Adjoint(TwoModeSum)", adjoint_rotation)
qml.add_decomps("Pow(TwoModeSum)", pow_rotation)
qml.add_decomps("qCond(TwoModeSum)", to_native_qcond(1))


class Gaussian(CVOperation):
    r"""General multi-qumode gaussian operation

    This gate is specified by a symplectic matrix :math:`\mathbf{S}` that transforms the quadrature
    operators

    .. math::

        \begin{pmatrix} \mathbf{x}' = \mathbf{S}\mathbf{x} \end{pmatrix}

    where :math:`\mathbf{x} = (1~x_1~p_1\dots x_n~p_n)^T`. Note that compared to the notation of eq. 101 [1]_,
    we instead incorporate the displacement term into an extra homogenous coordinate.

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 1

    resource_keys = {"num_wires"}

    def __init__(self, s: TensorLike, wires: WiresLike, id: str | None = None):
        r"""
        Args:
            s: A symplectic matrix of shape ``(2n+1, 2n+1)``, where ``n`` is the number of wires

            wires: The wires this gate acts on

            id: An optional identifying label
        """
        shape = qml.math.shape(s)

        if shape[0] % 2 != 1:
            raise ValueError("The symplectic matrix is missing a homogenous coordinate")

        if not is_symplectic(s):
            raise ValueError("The Gaussian gate requires a symplectic matrix")

        wires = Wires(wires)
        if shape[0] // 2 != len(wires):
            word = "few" if len(wires) < shape[0] // 2 else "many"
            raise ValueError(
                f"The gate was given too {word} wires for the input matrix"
            )

        self.hyperparameters["num_wires"] = len(wires)
        super().__init__(s, wires=wires, id=id)

    @property
    @override
    def ndim_params(self):
        return (2,)

    @property
    @override
    def num_wires(self):
        return len(self._wires)

    @property
    def resource_params(self):
        return {"num_wires": self.hyperparameters["num_wires"]}

    @override
    def adjoint(self):
        # Haven't found a `qml.math.inv` function, but we might still do it tensor-agnostic
        # using the array api
        from array_api_compat import array_namespace

        s = self.data[0]
        xp = array_namespace(s)
        sinv = xp.linalg.inv(s)
        return Gaussian(sinv, self.wires)

    @override
    def simplify(self):
        s = self.data[0]
        shape = qml.math.shape(s)

        if _can_replace(s, qml.math.eye(*shape, like=s)):
            return qml.Identity(self.wires)

        return self

    @override
    @staticmethod
    def _heisenberg_rep(p):
        return p[0]

    @override
    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "G", cache=cache
        )


def _can_replace(x, y):
    return (
        not qml.math.is_abstract(x)
        and not qml.math.requires_grad(x)
        and qml.math.allclose(x, y)
    )
