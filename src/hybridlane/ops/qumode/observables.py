# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math
from collections.abc import Iterable, Sequence
from typing import Any, Hashable

import numpy as np
import pennylane as qp
from pennylane.operation import Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

import hybridlane as hl

from ..mixins import FockRepresentation, Spectral
from .parametric_ops_single_qumode import Rotation


class QuadX(qp.QuadX, Spectral, FockRepresentation):
    r"""Position operator :math:`\hat{x}`

    The continuous-variable position operator is defined by its action on position
    "eigenstates" :math:`\ket{x}` as

    .. math::

        \hat{x} \ket{x} = x \ket{x}.

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 0
    * Number of dimensions per parameter: None

    It has a representation in the Fock basis defined in standard units with :math:`\hbar = 1`

    >>> X.compute_fock_matrix((5,))
    array([[0.    , 0.7071, 0.    , 0.    , 0.    ],
           [0.7071, 0.    , 1.    , 0.    , 0.    ],
           [0.    , 1.    , 0.    , 1.2247, 0.    ],
           [0.    , 0.    , 1.2247, 0.    , 1.4142],
           [0.    , 0.    , 0.    , 1.4142, 0.    ]])
    """

    @property
    def natural_basis(self):
        return hl.sa.ComputationalBasis.Position

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[Operator]:
        return []

    def position_spectrum(self, *basis_states) -> Sequence[float]:
        return basis_states[0]

    def __repr__(self):
        inner = ", ".join(map(str, self.wires))
        return f"x̂({inner})"

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...]) -> np.ndarray:
        # Standard units
        lam = 1 / math.sqrt(2)
        a = hl.CreationOp.compute_fock_matrix(wire_dims)
        ad = hl.AnnihilationOp.compute_fock_matrix(wire_dims)
        return lam * (a + ad)


X = QuadX
r"""Position operator :math:`\hat{x}`

.. seealso::

    This is an alias for :class:`~hybridlane.QuadX`
"""


class QuadP(qp.QuadP, Spectral, FockRepresentation):
    r"""Momentum operator :math:`\hat{p}`

    The continuous-variable momentum operator is defined by its action on momentum
    "eigenstates" :math:`\ket{p}` as

    .. math::

        \hat{p} \ket{p} = p \ket{p}.

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 0
    * Number of dimensions per parameter: None

    It has a representation in the Fock basis defined in standard units with :math:`\hbar = 1`

    >>> P.compute_fock_matrix((5,))
    array([[0.+0.j    , 0.+0.7071j, 0.+0.j    , 0.+0.j    , 0.+0.j    ],
           [0.-0.7071j, 0.+0.j    , 0.+1.j    , 0.+0.j    , 0.+0.j    ],
           [0.+0.j    , 0.-1.j    , 0.+0.j    , 0.+1.2247j, 0.+0.j    ],
           [0.+0.j    , 0.+0.j    , 0.-1.2247j, 0.+0.j    , 0.+1.4142j],
           [0.+0.j    , 0.+0.j    , 0.+0.j    , 0.-1.4142j, 0.+0.j    ]])
    """

    @property
    def natural_basis(self):
        return hl.sa.ComputationalBasis.Position

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[Operator]:
        return [Rotation(math.pi / 2, wires=wires)]  # rotate p -> x

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires | Iterable[Hashable] | Hashable | None = None,
        **hyperparameters: dict[str, Any],
    ) -> list[Operator]:
        return [Rotation(-math.pi / 2, wires=wires), QuadX(wires)]

    def __repr__(self):
        inner = ", ".join(map(str, self.wires))
        return f"p̂({inner})"

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...]) -> np.ndarray:
        # Standard units
        lam = 1 / math.sqrt(2)
        a = hl.CreationOp.compute_fock_matrix(wire_dims)
        ad = hl.AnnihilationOp.compute_fock_matrix(wire_dims)
        return lam * -1j * (a - ad)


P = QuadP
r"""Momentum operator :math:`\hat{p}`

.. seealso::

    This is an alias for :class:`~hybridlane.QuadP`
"""


class QuadOperator(qp.QuadOperator, Spectral, FockRepresentation):
    r"""The generalized quadrature observable :math:`\hat{x}_\phi = \hat{x} \cos\phi + \hat{p} \sin\phi`

    When used with the :func:`~hybridlane.expval` function, the expectation
    value :math:`\braket{\hat{x_\phi}}` is returned. This corresponds to
    the mean displacement in the phase space along axis at angle :math:`\phi`.

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: ``(0,)``

    Its representation in the Fock basis defined in standard units with :math:`\hbar = 1` is

    >>> QuadOperator.compute_fock_matrix((3,), np.pi / 4)
    array([[0.    +0.j    , 0.5   +0.5j   , 0.    +0.j    ],
           [0.5   -0.5j   , 0.    +0.j    , 0.7071+0.7071j],
           [0.    +0.j    , 0.7071-0.7071j, 0.    +0.j    ]])
    """

    @property
    def natural_basis(self):
        return hl.sa.ComputationalBasis.Position

    @staticmethod
    def compute_diagonalizing_gates(
        *params: TensorLike,
        wires: Wires | Iterable[Hashable] | Hashable,
        **hyperparams: dict[str, Any],
    ) -> list[Operator]:
        return [Rotation(params[0], wires)]

    @staticmethod
    def compute_decomposition(
        *params: TensorLike,
        wires: Wires | Iterable[Hashable] | Hashable | None = None,
        **hyperparameters: dict[str, Any],
    ) -> list[Operator]:
        return [Rotation(params[0], wires=wires), QuadX(wires)]

    def __repr__(self):
        inner = ", ".join(map(str, self.wires))
        return f"x̂_ϕ({inner})"

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], phi) -> TensorLike:
        x = QuadX.compute_fock_matrix(wire_dims)
        x = hl.math.asarray(x, like=phi)
        p = QuadP.compute_fock_matrix(wire_dims)
        p = hl.math.asarray(p, like=phi)
        return x * hl.math.cos(phi) + p * hl.math.sin(phi)


class NumberOperator(qp.NumberOperator, Spectral, FockRepresentation):
    r"""Number operator :math:`\hat{n}`

    The number operator is defined by its action on Fock states :math:`\ket{n}` as

    .. math::

        \hat{n} \ket{n} = n \ket{n}.

    When used with the :func:`~hybridlane.expval` function, the expectation value
    :math:`\braket{\hat{n}}` is returned. This corresponds to the mean photon number in the
    mode.

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 0
    * Number of dimensions per parameter: None

    It has a diagonal representation in the Fock basis

    >>> N.compute_fock_matrix((5,))
    array([[0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 2, 0, 0],
           [0, 0, 0, 3, 0],
           [0, 0, 0, 0, 4]])
    """

    @property
    def natural_basis(self):
        return hl.sa.ComputationalBasis.Discrete

    @staticmethod
    def compute_diagonalizing_gates(wires: WiresLike) -> list[Operator]:
        return []

    def fock_spectrum(self, *basis_states) -> Sequence[float]:
        return basis_states[0]

    def __repr__(self):
        inner = ", ".join(map(str, self.wires))
        return f"n̂({inner})"

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...]) -> np.ndarray:
        return hl.math.diag(hl.math.arange(wire_dims[0]))


N = NumberOperator
r"""Number operator :math:`\hat{n}`

.. seealso::

    This is an alias for :class:`~hybridlane.NumberOperator`
"""


class FockStateProjector(qp.FockStateProjector, Spectral, FockRepresentation):
    r"""The projector onto a multi-mode Fock state :math:`\ket{n_1, n_2, \ldots, n_m}\bra{n_1, n_2, \ldots, n_m}`

    When used with the :func:`~hybridlane.expval` function, the expectation value
    :math:`\braket{\ket{n_1, n_2, \ldots, n_m}\bra{n_1, n_2, \ldots, n_m}}` is returned. This corresponds to the probability of the system being in the Fock state :math:`\ket{n_1, n_2, \ldots, n_m}`.

    **Details**:

    * Number of wires: Any
    * Wire arguments: ``[qumode, ...]``
    * Number of parameters: Any
    * Number of dimensions per parameter: 0

    The number of parameters must match the number of wires, with each parameter being an
    integer. For example, to measure the probability of a single mode being in state
    :math:`\ket{3}`, the operator :math:`\ket{3}\bra{3}` would be instantiated as

    >>> op = FockStateProjector(3, wires=0)

    The corresponding matrix representation in the Fock basis would then be

    >>> op.fock_matrix({0: 5})
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0.]])


    The multi-mode operator :math:`\ket{1, 2}\bra{1, 2}` would be instantiated as

    >>> op = FockStateProjector([1, 2], wires=[0, 1])

    with the corresponding matrix representation in the Fock basis given by

    >>> op.fock_matrix({0: 3, 1: 3})
    array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    """

    @property
    def natural_basis(self):
        return hl.sa.ComputationalBasis.Discrete

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_diagonalizing_gates(
        *parameters: TensorLike, wires: WiresLike, **hyperparameters
    ) -> list[Operator]:
        return []

    def fock_spectrum(self, *basis_states) -> Sequence[float]:
        n = hl.math.reshape(self.data[0], (len(self.wires), -1))
        basis_states = hl.math.asarray(basis_states, like=n)  # (w, n?)
        basis_states = hl.math.reshape(
            basis_states, (len(self.wires), -1)
        )  # add n if necessary -> (w, n)
        row_matches = hl.math.all(n == basis_states, axis=0)  # -> (n,)
        return row_matches + 0

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], n) -> TensorLike:
        n = hl.math.asarray(n)

        if n.ndim == 0:
            n = hl.math.reshape(n, (1,))

        wire_dims = hl.math.asarray(wire_dims, like=n)
        if hl.math.any(n >= wire_dims):
            raise ValueError(
                f"Fock state projector cannot be constructed for n={n} with dimension {wire_dims}"
            )

        dim = int(hl.math.prod(wire_dims))
        proj = hl.math.zeros((dim, dim))
        idx = hl.math.ravel_multi_index(n, wire_dims)
        proj[idx, idx] = 1
        return hl.math.asarray(proj, like=n)
