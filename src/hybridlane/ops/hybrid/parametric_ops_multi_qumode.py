# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: D107, D102
r"""Hybrid CV-DV operations acting on multiple qumodes"""

import math
from typing import ClassVar

import pennylane as qp
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    pow_rotation,
)
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

import hybridlane as hl

from ...math.utils import can_replace, concrete_or_error
from ..mixins import FockRepresentation, HybridOperation
from ..op_math.decompositions.qubit_conditioned_decompositions import (
    decompose_multiqcond_native,
)
from ..qumode import Beamsplitter, TwoModeSqueezing
from .non_parametric_ops import ConditionalParity


class ConditionalBeamsplitter(HybridOperation, FockRepresentation):
    r"""Qubit-conditioned beamsplitter :math:`CBS(\theta, \varphi)`

    .. math::

        CBS(\theta, \varphi) &= \exp[-i\frac{\theta}{2}\sigma_z (e^{i\varphi}\ad b
                                + e^{-i\varphi} ab^\dagger)] \\
                             &= \begin{pmatrix}
                                    BS(\theta, \varphi) & 0 \\
                                    0 & BS^\dagger(\theta, \varphi)
                                \end{pmatrix}

    where :math:`\theta \in [0, 4\pi)` and :math:`\varphi \in [0, \pi)` (Table III.3 of
    :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 3
    * Wire arguments: ``[qubit, qumode, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    This is the qubit-conditioned version of the :class:`~hybridlane.ops.Beamsplitter` gate.

    >>> hl.qcond(hl.BS(0.5, 0.25, wires=[1, 2]), control_wires=0)
    ConditionalBeamsplitter(0.5, 0.25, wires=[0, 1, 2])

    There exists a decomposition in terms of :class:`.ConditionalParity` and
    :class:`~hybridlane.ops.Beamsplitter` gates (eq. 19 of :footcite:p:`crane2024hybrid`)

    .. math::

        CBS_{ijk}(\theta, \varphi) = CP_{ij}~BS_{jk}(\theta, \varphi + \pi/2)~
            CP_{ij}^\dagger

    .. seealso::

        :py:class:`~hybridlane.ops.Beamsplitter`

    References:
    ----------

    .. footbibliography::
    """

    num_params = 2
    num_wires = 3
    num_qumodes = 2
    ndim_params = (0, 0)

    resource_keys: ClassVar = set()

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(theta, phi, wires=wires, id=id)

    def adjoint(self):
        return ConditionalBeamsplitter(-self.data[0], self.data[1], self.wires)  # ty:ignore[unsupported-operator]

    def pow(self, z: int | float):
        return [ConditionalBeamsplitter(self.data[0] * z, self.data[1], self.wires)]  # ty:ignore[unsupported-operator]

    def simplify(self):
        theta = self.data[0] % (4 * math.pi)  # ty:ignore[unsupported-operator]
        phi = self.data[1] % math.pi  # ty:ignore[unsupported-operator]

        theta = concrete_or_error(None, theta, "Cannot simplify CBS when ``theta`` is a tracer")
        if can_replace(theta, 0):
            return qp.Identity(self.wires)

        return ConditionalBeamsplitter(theta, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "CBS", cache=cache)

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], theta, phi) -> TensorLike:  # ty:ignore[invalid-method-override]
        bs = hl.BS.compute_fock_matrix(wire_dims[1:], theta, phi)
        bsd = hl.math.conj(hl.math.transpose(bs))
        return hl.math.block_diag([bs, bsd])


@qp.register_resources({Beamsplitter: 1, ConditionalParity: 2})
def _cbs_parity_decomp(theta, phi, wires, **_):
    qp.adjoint(ConditionalParity)(wires[:2])
    Beamsplitter(theta, phi + math.pi / 2, wires[1:])
    ConditionalParity(wires[:2])


@qp.register_resources({ConditionalBeamsplitter: 1})
def _pow_cbs(theta, phi, wires, z, **_):
    ConditionalBeamsplitter(theta * z, phi, wires)


qp.add_decomps(ConditionalBeamsplitter, _cbs_parity_decomp)
qp.add_decomps("Adjoint(ConditionalBeamsplitter)", adjoint_rotation)
qp.add_decomps("Pow(ConditionalBeamsplitter)", _pow_cbs)
qp.add_decomps("qCond(ConditionalBeamsplitter)", decompose_multiqcond_native)

CBS = ConditionalBeamsplitter
r"""Qubit-conditioned beamsplitter :math:`CBS(\theta, \varphi)`

.. math::

    CBS(\theta, \varphi) = \exp[-i\frac{\theta}{2}\sigma_z (e^{i\varphi}\ad b
                            + e^{-i\varphi} ab^\dagger)]

.. seealso::

    This is an alias for :class:`~hybridlane.ConditionalBeamsplitter`
"""


class ConditionalTwoModeSqueezing(HybridOperation, FockRepresentation):
    r"""Qubit-conditioned two-mode squeezing :math:`CTMS(\xi)`

    .. math::

        CTMS(\xi) &= \exp[\sigma_z (\xi \ad b^\dagger - \xi^* ab)] \\
                    &= \begin{pmatrix}
                        TMS(\xi) & 0 \\
                        0 & TMS^\dagger(\xi)
                    \end{pmatrix}

    where :math:`\xi = re^{i\phi} \in \mathbb{C}` (Table III.3 of
    :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 3
    * Wire arguments: ``[qubit, qumode, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    This is the qubit-conditioned version of the :class:`~hybridlane.ops.TwoModeSqueezing` gate.

    >>> hl.qcond(hl.TMS(0.5, 0.25, wires=[1, 2]), control_wires=0)
    ConditionalTwoModeSqueezing(0.5, 0.25, wires=[0, 1, 2])

    There exists a decomposition in terms of :class:`.ConditionalParity` and
    :class:`~hybridlane.ops.TwoModeSqueezing` gates (eq. 20 of :footcite:p:`crane2024hybrid`)

    .. math::

        CTMS_{ijk}(\xi) = CP_{ij} TMS_{jk}(i\xi) CP_{ij}^\dagger

    .. seealso::

        :py:class:`~hybridlane.ops.TwoModeSqueezing`

    References:
    ----------

    .. footbibliography::
    """

    num_params = 2
    num_wires = 3
    num_qumodes = 2
    ndim_params = (0, 0)

    resource_keys: ClassVar = set()

    def __init__(
        self,
        r: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(r, phi, wires=wires, id=id)

    def pow(self, z: int | float):
        r, phi = self.data
        return [ConditionalTwoModeSqueezing(r * z, phi, self.wires)]  # ty:ignore[unsupported-operator]

    def adjoint(self):
        return [ConditionalTwoModeSqueezing(-self.data[0], self.data[1], self.wires)]  # ty:ignore[unsupported-operator]

    def simplify(self):
        r, phi = self.data[0], self.data[1] % (2 * math.pi)  # ty:ignore[unsupported-operator]

        r = concrete_or_error(None, r, "Cannot simplify CTMS when ``r`` is a tracer")
        if can_replace(r, 0):
            return qp.Identity(self.wires)

        return ConditionalTwoModeSqueezing(r, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "CTMS", cache=cache)

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], r, phi) -> TensorLike:  # ty:ignore[invalid-method-override]
        tms = hl.TMS.compute_fock_matrix(wire_dims[1:], r, phi)
        tmsd = hl.math.conj(hl.math.transpose(tms))
        return hl.math.block_diag([tms, tmsd])


@qp.register_resources({TwoModeSqueezing: 1, ConditionalParity: 2})
def _ctms_parity_decomp(r, phi, wires, **_):
    qp.adjoint(ConditionalParity)(wires[:2])
    TwoModeSqueezing(r, phi + math.pi / 2, wires[1:])
    ConditionalParity(wires[:2])


@qp.register_resources({ConditionalTwoModeSqueezing: 1})
def _pow_ctms(theta, phi, wires, z, **_):
    ConditionalTwoModeSqueezing(theta * z, phi, wires)


qp.add_decomps(ConditionalTwoModeSqueezing, _ctms_parity_decomp)
qp.add_decomps("Adjoint(ConditionalTwoModeSqueezing)", adjoint_rotation)
qp.add_decomps("Pow(ConditionalTwoModeSqueezing)", _pow_ctms)
qp.add_decomps("qCond(ConditionalTwoModeSqueezing)", decompose_multiqcond_native)

CTMS = ConditionalTwoModeSqueezing
r"""Qubit-conditioned two-mode squeezing :math:`CTMS(\xi)`

.. math::

    CTMS(\xi) = \exp[\sigma_z (\xi \ad b^\dagger - \xi^* ab)]

.. seealso::

    This is an alias for :class:`~hybridlane.ConditionalTwoModeSqueezing`
"""


class ConditionalTwoModeSum(HybridOperation, FockRepresentation):
    r"""Qubit-conditioned two-mode sum gate :math:`CSUM(\lambda)`

    .. math::

        CSUM(\lambda) &= \exp[\frac{\lambda}{2}\sigma_z(a + \ad)(b^\dagger - b)] \\
                        &= \begin{pmatrix}
                            SUM(\lambda) & 0 \\
                            0 & SUM^\dagger(\lambda)
                        \end{pmatrix}

    with :math:`\lambda \in \mathbb{R}` (Table III.3 of :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 3
    * Wire arguments: ``[qubit, qumode, qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: ``(0,)``

    This is the qubit-conditioned version of the :class:`~hybridlane.ops.TwoModeSum` gate.

    >>> hl.qcond(hl.SUM(0.5, wires=[1, 2]), control_wires=0)
    ConditionalTwoModeSum(0.5, wires=[0, 1, 2])

    .. seealso::

        :py:class:`~hybridlane.ops.TwoModeSum`

    References:
    ----------

    .. footbibliography::
    """

    num_params = 1
    num_wires = 3
    num_qumodes = 2
    ndim_params = (0,)

    resource_keys: ClassVar = set()

    def __init__(self, lam: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(lam, wires=wires, id=id)

    def adjoint(self):
        lambda_ = self.parameters[0]
        return ConditionalTwoModeSum(-lambda_, wires=self.wires)  # ty:ignore[unsupported-operator]

    def pow(self, z: int | float):
        return [ConditionalTwoModeSum(self.data[0] * z, self.wires)]  # ty:ignore[unsupported-operator]

    def simplify(self):
        lambda_ = self.data[0]

        lambda_ = concrete_or_error(
            None, lambda_, "Cannot simplify CSUM when ``lambda`` is a tracer"
        )
        if can_replace(lambda_, 0):
            return qp.Identity(self.wires)

        return ConditionalTwoModeSum(lambda_, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "CSUM", cache=cache)

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], lam) -> TensorLike:  # ty:ignore[invalid-method-override]
        tms = hl.SUM.compute_fock_matrix(wire_dims[1:], lam)
        tmsd = hl.math.conj(hl.math.transpose(tms))
        return hl.math.block_diag([tms, tmsd])


qp.add_decomps("Adjoint(ConditionalTwoModeSum)", adjoint_rotation)
qp.add_decomps("Pow(ConditionalTwoModeSum)", pow_rotation)
qp.add_decomps("qCond(ConditionalTwoModeSum)", decompose_multiqcond_native)

CSUM = ConditionalTwoModeSum
r"""Qubit-conditioned two-mode sum gate :math:`CSUM(\lambda)`

.. math::

    CSUM(\lambda) = \exp[\frac{\lambda}{2}\sigma_z(a + \ad)(b^\dagger - b)]

.. seealso::

    This is an alias for :class:`~hybridlane.ConditionalTwoModeSum`
"""
