# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import numpy as np
import pennylane as qp
from pennylane.decomposition.symbolic_decomposition import (
    make_pow_decomp_with_period,
    pow_involutory,
    self_adjoint,
)
from pennylane.operation import CV, CVOperation, Operator
from pennylane.wires import WiresLike

import hybridlane as hl

from ..mixins import FockRepresentation
from ..op_math.decompositions.qubit_conditioned_decompositions import to_native_qcond


class Fourier(CVOperation, FockRepresentation):
    r"""Continuous-variable Fourier gate :math:`F`

    This gate is a special case of the CV :py:class:`~hybridlane.Rotation` gate with
    :math:`\theta = \pi/2`

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 0
    * Number of dimensions per parameter: None

    Its symplectic representation is given (in standard units) by

    .. math::

        \begin{pmatrix}
            I \\
            \hat{x}' \\
            \hat{p}'
        \end{pmatrix} =
        \begin{pmatrix}
            1 & 0 & 0 \\
            0 & 0 & 1 \\
            0 & -1 & 0
        \end{pmatrix}
        \begin{pmatrix}
            I \\
            \hat{x} \\
            \hat{p}
        \end{pmatrix}

    For specific parameter values, it may be obtained like

    >>> F(wires=0).heisenberg_tr((0,))
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0., -1.,  0.]])
    """

    num_params = 0
    num_wires = 1

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        return hl.math.symplectic.rotation(math.pi / 2)

    def adjoint(self):
        return hl.Rotation(-math.pi / 2, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "F", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...]) -> np.ndarray:
        n = hl.math.arange(wire_dims[0])
        return hl.math.diag(hl.math.exp(-1j * math.pi / 2 * n))


def _f_to_r_resources():
    return {hl.Rotation: 1}


@qp.register_resources(_f_to_r_resources)
def _f_to_r(wires, **_):
    hl.Rotation(math.pi / 2, wires)


def _adjoint_f_to_r_resources():
    return {hl.Rotation: 1}


@qp.register_resources(_adjoint_f_to_r_resources)
def _adjoint_f_to_r(wires, **_):
    hl.Rotation(-math.pi / 2, wires)


def _pow_f_to_r_resources(z, **_):
    return {hl.Rotation: 1}


@qp.register_resources(_pow_f_to_r_resources)
def _pow_f_to_r(wires, z, **_):
    hl.Rotation(math.pi / 2 * z, wires)


qp.add_decomps(Fourier, _f_to_r)
qp.add_decomps("Adjoint(Fourier)", _adjoint_f_to_r)
qp.add_decomps("Pow(Fourier)", make_pow_decomp_with_period(4), _pow_f_to_r)
qp.add_decomps("qCond(Fourier)", to_native_qcond(1))

F = Fourier
r"""Fourier gate

.. math::

    F = e^{-i\frac{\pi}{2}\hat{n}}

.. seealso::

    This is an alias of :class:`~hybridlane.Fourier`
"""


class ModeSwap(CVOperation, FockRepresentation):
    r"""Continuous-variable SWAP between two qumodes

    This has a decomposition in terms of a :py:class:`~hybridlane.Beamsplitter` and
    phase-space :py:class:`~hybridlane.Rotation` gates to eliminate the global phase.
    (eq. 175 of :footcite:p:`liu2026hybrid`):

    .. math::

        \text{ModeSwap} = R_1(-\pi/2) R_2(-\pi/2) BS(\pi, 0)

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qumode, qumode]``
    * Number of parameters: 0
    * Number of dimensions per parameter: None

    Its symplectic representation is the permutation

    .. math::

        \begin{pmatrix}
            I \\
            \hat{x}_a' \\
            \hat{p}_a' \\
            \hat{x}_b' \\
            \hat{p}_b'
        \end{pmatrix} =
        \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 1 \\
            0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0
        \end{pmatrix}
        \begin{pmatrix}
            I \\
            \hat{x}_a \\
            \hat{p}_a \\
            \hat{x}_b \\
            \hat{p}_b
        \end{pmatrix}

    It has a matrix representation in the Fock basis

    >>> ModeSwap((0, 1)).fock_matrix({0: 2, 1: 2})
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]])

    which for dimension 2 on each wire corresponds to the familiar qubit SWAP gate.

    Directly obtaining the matrix from ``ModeSwap`` as opposed to from the decomposition is
    more accurate due to truncation effects in the :math:`BS` gate (note the phase of
    :math:`-1` on the element :math:`\ket{1, 1} \to \ket{1, 1}` in the example below):

    >>> r = hl.R(-np.pi/2, wires=0).fock_matrix({0: 2})
    >>> bs = hl.BS(np.pi, 0, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    >>> hl.math.kron(r, r) @ bs
    array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  1.-0.j,  0.+0.j],
           [ 0.+0.j,  1.-0.j,  0.+0.j,  0.+0.j],
           [ 0.+0.j,  0.+0.j,  0.+0.j, -1.+0.j]])

    References
    ----------

    .. footbibliography::
    """

    num_params = 0
    num_wires = 2

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(_):
        return np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=float,
        )

    def adjoint(self):
        return ModeSwap(self.wires)  # self-adjoint up to a global phase of -1

    def pow(self, z: int | float):
        if isinstance(z, float):
            raise NotImplementedError("Unknown formula for fractional powers")
        elif z < 0:
            raise NotImplementedError("Unknown formula for inverse")

        if z % 2 == 0:
            return [qp.Identity(self.wires)]
        else:
            return [ModeSwap(self.wires)]

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...]) -> np.ndarray:
        dim0, dim1 = wire_dims
        mat = hl.math.zeros((dim0 * dim1, dim0 * dim1), dtype=complex)
        m = hl.math.arange(dim0)[:, None]
        n = hl.math.arange(dim1)[None, :]
        row_indices = n + (m * dim1)
        col_indices = (n * dim0) + m
        mat[row_indices.flatten(), col_indices.flatten()] = 1
        return mat


def _swap_to_bs_resources():
    return {
        hl.Beamsplitter: 1,
        hl.Rotation: 2,
    }


@qp.register_resources(_swap_to_bs_resources)
def _swap_to_bs(wires, **_):
    hl.Beamsplitter(math.pi, 0, wires)
    hl.Rotation(-math.pi / 2, wires[0])
    hl.Rotation(-math.pi / 2, wires[1])


qp.add_decomps(ModeSwap, _swap_to_bs)
qp.add_decomps("Adjoint(ModeSwap)", self_adjoint)
qp.add_decomps("Pow(ModeSwap)", pow_involutory)


class CreationOp(CV, Operator, FockRepresentation):
    r"""Continuous-variable creation operator :math:`\ad`

    .. math::

        \ad \ket{n} = \sqrt{n+1} \ket{n+1}

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 0
    * Number of dimensions per parameter: None

    This operator is not Hermitian and so cannot be used as an observable.

    >>> CreationOp(0).is_verified_hermitian
    False

    It has a matrix representation in the Fock basis:

    >>> CreationOp.compute_fock_matrix((5,))
    array([[0.    , 0.    , 0.    , 0.    , 0.    ],
           [1.    , 0.    , 0.    , 0.    , 0.    ],
           [0.    , 1.4142, 0.    , 0.    , 0.    ],
           [0.    , 0.    , 1.7321, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 2.    , 0.    ]])
    """

    num_params = 0
    num_wires = 1
    is_verified_hermitian = False

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    def adjoint(self):
        return AnnihilationOp(self.wires)

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...]) -> np.ndarray:
        return hl.math.diag([math.sqrt(i) for i in range(1, wire_dims[0])], k=-1)


Ad = CreationOp
r"""Creation operator :math:`\ad`

.. seealso::

    This is an alias of :class:`~hybridlane.CreationOp`
"""


class AnnihilationOp(CV, Operator, FockRepresentation):
    r"""Continuous-variable annihilation operator :math:`a`

    .. math::

        a \ket{n} = \sqrt{n} \ket{n-1}

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 0
    * Number of dimensions per parameter: None

    This operator is not Hermitian and so cannot be used as an observable:

    >>> AnnihilationOp(0).is_verified_hermitian
    False

    It has a matrix representation in the Fock basis:

    >>> AnnihilationOp.compute_fock_matrix((5,))
    array([[0.    , 1.    , 0.    , 0.    , 0.    ],
           [0.    , 0.    , 1.4142, 0.    , 0.    ],
           [0.    , 0.    , 0.    , 1.7321, 0.    ],
           [0.    , 0.    , 0.    , 0.    , 2.    ],
           [0.    , 0.    , 0.    , 0.    , 0.    ]])
    """

    num_params = 0
    num_wires = 1
    is_verified_hermitian = False

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    def adjoint(self):
        return CreationOp(self.wires)

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...]) -> np.ndarray:
        return hl.math.diag([math.sqrt(i) for i in range(1, wire_dims[0])], k=1)


A = AnnihilationOp
r"""Annihilation operator :math:`a`

.. seealso::

    This is an alias of :class:`~hybridlane.AnnihilationOp`
"""
