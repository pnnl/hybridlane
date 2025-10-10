# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math

import pennylane as qml
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    pow_rotation,
)
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from ..mixins import Hybrid
from ..qumode import Beamsplitter, TwoModeSqueezing
from .non_parametric_ops import ConditionalParity
from .parametric_ops_single_qumode import ConditionalRotation


class ConditionalBeamsplitter(Operation, Hybrid):
    r"""Qubit-conditioned beamsplitter :math:`CBS(\theta, \varphi)`

    This is a multi-qumode gate conditioned on a qubit. It is given by the expression

    .. math::

        CBS(\theta, \varphi) &= \exp[-i\frac{\theta}{2}\sigma_z (e^{i\varphi}\ad b + e^{-i\varphi} ab^\dagger)] \\
                             &= \ket{0}\bra{0} \otimes BS(\theta, \varphi) + \ket{1}\bra{1} \otimes BS(-\theta, \varphi)

    where :math:`\theta \in [0, 4\pi)` and :math:`\varphi \in [0, \pi)` (Table III.3 [1]_). There exists a decomposition in terms
    of :class:`.ConditionalParity` and :class:`~hybridlane.ops.Beamsplitter` gates (eq. 19 [2]_)

    .. math::

        CBS_{ijk}(\theta, \varphi) = CP_{ij} BS_{jk}(\theta, \varphi + \pi/2) CP_{ij}^\dagger 

    .. seealso::

        :py:class:`~hybridlane.ops.Beamsplitter`

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    .. [2] E. Crane et al, 2024. `arXiv:2409.03747 <https://arxiv.org/abs/2409.03747>`_
    """

    num_params = 2
    num_wires = 3
    num_qumodes = 2
    ndim_params = (0, 0)

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

    def adjoint(self):
        return ConditionalBeamsplitter(-self.data[0], self.data[1], self.wires)

    def pow(self, z: int | float):
        return [ConditionalBeamsplitter(self.data[0] * z, self.data[1], self.wires)]

    def simplify(self):
        theta = self.data[0] % (4 * math.pi)
        phi = self.data[1] % math.pi

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return ConditionalBeamsplitter(theta, phi, self.wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        theta, phi = params
        return [
            ConditionalParity(wires[:2]).adjoint(),
            Beamsplitter(theta, phi + math.pi / 2, wires[1:]),
            ConditionalParity(wires[:2]),
        ]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CBS", cache=cache
        )


@qml.register_resources({Beamsplitter: 1, ConditionalParity: 2})
def _cbs_parity_decomp(theta, phi, wires, **_):
    qml.adjoint(ConditionalParity)(wires[:2])
    Beamsplitter(theta, phi + math.pi / 2, wires[1:])
    ConditionalParity(wires[:2])


@qml.register_resources({ConditionalBeamsplitter: 1})
def _pow_cbs(theta, phi, wires, z, **_):
    ConditionalBeamsplitter(theta * z, phi, wires)


qml.add_decomps(ConditionalBeamsplitter, _cbs_parity_decomp)
qml.add_decomps("Adjoint(ConditionalBeamsplitter)", adjoint_rotation)
qml.add_decomps("Pow(ConditionalBeamsplitter)", _pow_cbs)


class ConditionalTwoModeSqueezing(Operation, Hybrid):
    r"""Qubit-conditioned two-mode squeezing :math:`CTMS(\xi)`

    This is the qubit-conditioned version of the :py:class:`~hybridlane.ops.TwoModeSqueezing` gate, given by

    .. math::

        CTMS(\xi) &= \exp[\sigma_z (\xi \ad b^\dagger - \xi^* ab)] \\
                  &= \ket{0}\bra{0} \otimes TMS(\xi) + \ket{1}\bra{1} \otimes TMS(-\xi)

    where :math:`\xi = re^{i\phi} \in \mathbb{C}` (Table III.3 [1]_). There exists a decomposition in terms of
    :class:`.ConditionalParity` and :class:`~hybridlane.ops.TwoModeSqueezing` gates (eq. 20 [2]_)

    .. math::

        CTMS_{ijk}(\xi) = CP_{ij} TMS_{jk}(i\xi) CP_{ij}^\dagger

    .. note::

        This formula differs from the Pennylane implementation by a minus sign (:math:`z \rightarrow -z`).

    .. seealso::

        :py:class:`~hybridlane.ops.TwoModeSqueezing`

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    .. [2] E. Crane et al, 2024. `arXiv:2409.03747 <https://arxiv.org/abs/2409.03747>`_
    """

    num_params = 2
    num_wires = 3
    num_qumodes = 2
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self,
        r: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(r, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def pow(self, z: int | float):
        r, phi = self.data
        return [ConditionalTwoModeSqueezing(r * z, phi, self.wires)]

    def adjoint(self):
        return [ConditionalTwoModeSqueezing(-self.data[0], self.data[1], self.wires)]

    def simplify(self):
        r, phi = self.data[0], self.data[1] % (2 * math.pi)

        if _can_replace(r, 0):
            return qml.Identity(self.wires)

        return ConditionalTwoModeSqueezing(r, phi, self.wires)

    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        r, phi = params
        return [
            ConditionalRotation(math.pi / 2, wires[:2]).adjoint(),
            TwoModeSqueezing(r, phi + math.pi / 2, wires[1:]),
            ConditionalRotation(math.pi / 2, wires[:2]),
        ]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CTMS", cache=cache
        )


@qml.register_resources({TwoModeSqueezing: 1, ConditionalParity: 2})
def _ctms_parity_decomp(r, phi, wires, **_):
    qml.adjoint(ConditionalParity)(wires[:2])
    TwoModeSqueezing(r, phi + math.pi / 2, wires[1:])
    ConditionalParity(wires[:2])


@qml.register_resources({ConditionalTwoModeSqueezing: 1})
def _pow_ctms(theta, phi, wires, z, **_):
    ConditionalTwoModeSqueezing(theta * z, phi, wires)


qml.add_decomps(ConditionalTwoModeSqueezing, _ctms_parity_decomp)
qml.add_decomps("Adjoint(ConditionalTwoModeSqueezing)", adjoint_rotation)
qml.add_decomps("Pow(ConditionalTwoModeSqueezing)", _pow_ctms)


class ConditionalTwoModeSum(Operation, Hybrid):
    r"""Qubit-conditioned two-mode sum gate :math:`CSUM(\lambda)`

    This is a multi-mode gate conditioned on the state of a qubit, given by the expression

    .. math::

        CSUM(\lambda) &= \exp[\frac{\lambda}{2}\sigma_z(a + \ad)(b^\dagger - b)] \\
                      &= \ket{0}\bra{0} \otimes SUM(\lambda) + \ket{1}\bra{1} \otimes SUM(-\lambda)

    with :math:`\lambda \in \mathbb{R}` (Table III.3 [1]_).

    .. seealso::

        :py:class:`~hybridlane.ops.TwoModeSum`

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 1
    num_wires = 3
    num_qumodes = 2
    ndim_params = (0,)

    resource_keys = set()

    def __init__(self, lam: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(lam, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def adjoint(self):
        lambda_ = self.parameters[0]
        return ConditionalTwoModeSum(-lambda_, wires=self.wires)

    def pow(self, z: int | float):
        return [ConditionalTwoModeSum(self.data[0] * z, self.wires)]

    def simplify(self):
        lambda_ = self.data[0]
        if _can_replace(lambda_, 0):
            return qml.Identity(self.wires)

        return ConditionalTwoModeSum(lambda_, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CSUM", cache=cache
        )


qml.add_decomps("Adjoint(ConditionalTwoModeSum)", adjoint_rotation)
qml.add_decomps("Pow(ConditionalTwoModeSum)", pow_rotation)


def _can_replace(x, y):
    return (
        not qml.math.is_abstract(x)
        and not qml.math.requires_grad(x)
        and qml.math.allclose(x, y)
    )
