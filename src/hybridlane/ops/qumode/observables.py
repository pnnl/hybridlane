# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math
from collections.abc import Iterable, Sequence
from typing import Any, Hashable

import numpy as np
import pennylane as qml
from pennylane.operation import Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

import hybridlane as hqml

from ..mixins import FockRepresentation, Spectral
from .parametric_ops_single_qumode import Rotation


class QuadX(qml.QuadX, Spectral, FockRepresentation):
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
    array([[0.        , 0.70710678, 0.        , 0.        , 0.        ],
           [0.70710678, 0.        , 1.        , 0.        , 0.        ],
           [0.        , 1.        , 0.        , 1.22474487, 0.        ],
           [0.        , 0.        , 1.22474487, 0.        , 1.41421356],
           [0.        , 0.        , 0.        , 1.41421356, 0.        ]])
    """

    @property
    def natural_basis(self):
        return hqml.sa.ComputationalBasis.Position

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
        a = hqml.CreationOp.compute_fock_matrix(wire_dims)
        ad = hqml.AnnihilationOp.compute_fock_matrix(wire_dims)
        return lam * (a + ad)


X = QuadX
r"""Position operator :math:`\hat{x}`

.. seealso::

    This is an alias for :class:`~hybridlane.QuadX`
"""


class QuadP(qml.QuadP, Spectral, FockRepresentation):
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

    >>> P.compute_fock_matrix((3,))
    array([[0.+0.j        , 0.+0.70710678j, 0.+0.j        ],
           [0.-0.70710678j, 0.+0.j        , 0.+1.j        ],
           [0.+0.j        , 0.-1.j        , 0.+0.j        ]])
    """

    @property
    def natural_basis(self):
        return hqml.sa.ComputationalBasis.Position

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
        a = hqml.CreationOp.compute_fock_matrix(wire_dims)
        ad = hqml.AnnihilationOp.compute_fock_matrix(wire_dims)
        return lam * -1j * (a - ad)


P = QuadP
r"""Momentum operator :math:`\hat{p}`

.. seealso::

    This is an alias for :class:`~hybridlane.QuadP`
"""


class QuadOperator(qml.QuadOperator, Spectral, FockRepresentation):
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

    .. invisible-code-block:: python

        import math

    >>> QuadOperator.compute_fock_matrix((3,), math.pi / 4)
    array([[0.        +0.j        , 0.5       +0.5j       ,
            0.        +0.j        ],
           [0.5       -0.5j       , 0.        +0.j        ,
            0.70710678+0.70710678j],
           [0.        +0.j        , 0.70710678-0.70710678j,
            0.        +0.j        ]])
    """

    @property
    def natural_basis(self):
        return hqml.sa.ComputationalBasis.Position

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
        x = hqml.math.asarray(x, like=phi)
        p = QuadP.compute_fock_matrix(wire_dims)
        p = hqml.math.asarray(p, like=phi)
        return x * hqml.math.cos(phi) + p * hqml.math.sin(phi)


class NumberOperator(qml.NumberOperator, Spectral, FockRepresentation):
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
        return hqml.sa.ComputationalBasis.Discrete

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
        return hqml.math.diag(hqml.math.arange(wire_dims[0]))


N = NumberOperator
r"""Number operator :math:`\hat{n}`

.. seealso::

    This is an alias for :class:`~hybridlane.NumberOperator`
"""


class FockStateProjector(qml.FockStateProjector, Spectral, FockRepresentation):
    @property
    def natural_basis(self):
        return hqml.sa.ComputationalBasis.Discrete

    @property
    def num_wires(self):
        return len(self.wires)

    @staticmethod
    def compute_diagonalizing_gates(
        *parameters: TensorLike, wires: WiresLike, **hyperparameters
    ) -> list[Operator]:
        return []

    def fock_spectrum(self, *basis_states) -> Sequence[float]:
        n = hqml.math.reshape(self.data[0], (1, -1))
        basis_states = hqml.math.asarray(basis_states, like=n)
        basis_states = hqml.math.reshape(basis_states, (-1, len(self.wires)))
        row_matches = hqml.math.all(n == basis_states, axis=-1)
        return row_matches + 0

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], n) -> TensorLike:
        wire_dims = hqml.math.asarray(wire_dims, like=n)
        n = hqml.math.asarray(n)
        if hqml.math.any(n >= wire_dims):
            raise ValueError(
                f"Fock state projector cannot be constructed for n={n} with dimension {wire_dims}"
            )

        dim = int(hqml.math.prod(wire_dims))
        proj = hqml.math.zeros((dim, dim))
        idx = hqml.math.ravel_multi_index(n, wire_dims)
        proj[idx, idx] = 1
        return hqml.math.asarray(proj, like=n)
