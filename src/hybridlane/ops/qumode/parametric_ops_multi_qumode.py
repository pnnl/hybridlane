# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import numpy as np
import pennylane as qp
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    pow_rotation,
)
from pennylane.operation import CVOperation
from pennylane.ops.cv import _rotation, _two_term_shift_rule
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

import hybridlane as hl

from ..mixins import FockRepresentation
from ..op_math.decompositions.qubit_conditioned_decompositions import to_native_qcond


# Change to match convention
class Beamsplitter(CVOperation, FockRepresentation):
    r"""Beamsplitter gate :math:`BS(\theta, \varphi)`

    .. math::

        BS(\theta,\varphi) = \exp\left[-i \frac{\theta}{2} (e^{i\varphi} \ad b
            + e^{-i\varphi}ab^\dagger)\right]

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qumode, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)```

    The beamsplitter gate conserves total excitation number, as :math:`[BS, n_a + n_b] = 0`.
    Its representation in the Fock basis can be obtained with:

    >>> BS(0.5, 0.1, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[ 1.    +0.j    ,  0.    +0.j    ,  0.    +0.j    ,
             0.    +0.j    ],
           [ 0.    +0.j    ,  0.9689-0.j    , -0.0247-0.2462j,
             0.    +0.j    ],
           [ 0.    +0.j    ,  0.0247-0.2462j,  0.9689+0.j    ,
             0.    +0.j    ],
           [ 0.    +0.j    ,  0.    +0.j    ,  0.    +0.j    ,
             1.    +0.j    ]])
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
            return qp.Identity(wires=self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "BS", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], theta, phi) -> TensorLike:
        dim_a, dim_b = wire_dims
        ad = hl.math.asarray(hl.CreationOp.compute_fock_matrix((dim_a,)), like=theta)
        b = hl.math.asarray(hl.AnnihilationOp.compute_fock_matrix((dim_b,)), like=theta)
        adb = hl.math.kron(ad, b)
        abd = hl.math.conj(hl.math.transpose(adb))
        g = -0.5j * theta * (hl.math.exp(1j * phi) * adb + hl.math.exp(-1j * phi) * abd)
        return hl.math.expm(g)


@qp.register_resources({Beamsplitter: 1})
def _pow_bs(theta, phi, wires, z, **_):
    Beamsplitter(theta * z, phi, wires)


qp.add_decomps("Adjoint(Beamsplitter)", adjoint_rotation)
qp.add_decomps("Pow(Beamsplitter)", _pow_bs)
qp.add_decomps("qCond(Beamsplitter)", to_native_qcond(1))

BS = Beamsplitter
r"""Beamsplitter gate :math:`BS(\theta, \varphi)`

.. math::

    BS(\theta, \varphi) = \exp\left[-i \frac{\theta}{2} (e^{i\varphi} \ad b
        + e^{-i\varphi}ab^\dagger)\right]

.. seealso::

    This is an alias of :class:`~hybridlane.Beamsplitter`
"""


# Re-export flipping sign of r, equivalent to φ -> φ + π
class TwoModeSqueezing(CVOperation, FockRepresentation):
    r"""Phase space two-mode squeezing :math:`TMS(r, \varphi)`

    .. math::

        TMS(r, \varphi) = \exp\left[r (e^{i\phi} \ad b^\dagger - e^{-i\phi} ab\right].

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qumode, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``
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
            return qp.Identity(self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "TMS", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], r, phi) -> TensorLike:
        ad = hl.math.asarray(hl.CreationOp.compute_fock_matrix(wire_dims[:1]), like=r)
        bd = hl.math.asarray(hl.CreationOp.compute_fock_matrix(wire_dims[1:]), like=r)
        adbd = hl.math.kron(ad, bd)
        ab = hl.math.conj(hl.math.transpose(adbd))
        g = r * (hl.math.exp(1j * phi) * adbd - hl.math.exp(-1j * phi) * ab)
        return hl.math.expm(g)


@qp.register_resources({TwoModeSqueezing: 1})
def _pow_tms(r, phi, wires, z, **_):
    TwoModeSqueezing(r * z, phi, wires)


qp.add_decomps("Adjoint(TwoModeSqueezing)", adjoint_rotation)
qp.add_decomps("Pow(TwoModeSqueezing)", _pow_tms)
qp.add_decomps("qCond(TwoModeSqueezing)", to_native_qcond(1))

TMS = TwoModeSqueezing
r"""Phase space two-mode squeezing :math:`TMS(r, \varphi)`

.. math::

    TMS(r, \varphi) = \exp\left[r (e^{i\phi} \ad b^\dagger - e^{-i\phi} ab\right].

.. seealso::

    This is an alias of :class:`~hybridlane.TwoModeSqueezing`
"""


class TwoModeSum(CVOperation, FockRepresentation):
    r"""Two-mode summing gate :math:`SUM(\lambda)`

    This continuous-variable gate implements the unitary

    .. math::

        SUM(\lambda) = \exp[\frac{\lambda}{2}(a + \ad)(b^\dagger - b)]

    where :math:`\lambda \in \mathbb{R}` is a real parameter.

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qumode, qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: ``(0,)``

    The action on the wavefunction is given by

    .. math::

        SUM(\lambda)\ket{x_a}\ket{x_b} = \ket{x_a}\ket{x_b + \lambda x_a}

    in the position basis (see Box III.6 of :footcite:p:`liu2026hybrid`).

    References
    ----------

    .. footbibliography::
    """

    num_params = 1
    num_wires = 2
    ndim_params = (0,)
    grad_method = "F"

    resource_keys = set()

    def __init__(self, lambda_: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(lambda_, wires=wires, id=id)

    def adjoint(self):
        lambda_ = self.parameters[0]
        return TwoModeSum(-lambda_, wires=self.wires)

    def pow(self, z: int | float):
        return [TwoModeSum(self.data[0] * z, self.wires)]

    def simplify(self):
        lambda_ = self.data[0]
        if _can_replace(lambda_, 0):
            return qp.Identity(self.wires)

        return TwoModeSum(lambda_, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "SUM", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], lambda_) -> TensorLike:
        a = hl.math.asarray(
            hl.AnnihilationOp.compute_fock_matrix(wire_dims[:1]), like=lambda_
        )
        ad = hl.math.conj(hl.math.transpose(a))
        b = hl.math.asarray(
            hl.AnnihilationOp.compute_fock_matrix(wire_dims[1:]), like=lambda_
        )
        bd = hl.math.conj(hl.math.transpose(b))
        g = 0.5 * lambda_ * hl.math.kron(a + ad, bd - b)
        return hl.math.expm(g)


qp.add_decomps("Adjoint(TwoModeSum)", adjoint_rotation)
qp.add_decomps("Pow(TwoModeSum)", pow_rotation)
qp.add_decomps("qCond(TwoModeSum)", to_native_qcond(1))

SUM = TwoModeSum
r"""Two-mode summing gate :math:`SUM(\lambda)`

.. math::

    SUM(\lambda) = \exp[\frac{\lambda}{2}(a + \ad)(b^\dagger - b)]

.. seealso::

    This is an alias of :class:`~hybridlane.TwoModeSum`
"""


def _can_replace(x, y):
    return (
        not qp.math.is_abstract(x)
        and not qp.math.requires_grad(x)
        and qp.math.allclose(x, y)
    )
