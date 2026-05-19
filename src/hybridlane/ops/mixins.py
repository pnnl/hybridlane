# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pennylane.exceptions import MatrixUndefinedError
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

import hybridlane as hqml


class Spectral:
    r"""Mixin for observables that have an infinite number of eigenvalues (spectrum)

    Instead of enumerating all eigenvalues like normal Pennylane observables (because that is no
    longer possible), this mixin provides a general framework for observables to define their spectrum,
    a function :math:`f: \mathcal{B} \rightarrow \mathbb{R}` from basis states to eigenvalues.
    """

    @property
    def natural_basis(self) -> hqml.sa.ComputationalBasis:
        raise NotImplementedError(
            "Observable did not define its best basis to measure in"
        )

    # todo: decide whether we should have dv spectrums too

    def position_spectrum(self, *basis_states: TensorLike) -> Sequence[float]:
        r"""Provides a diagonal decomposition of the operator in the position basis

        An observable that implements this method guarantees it can be written as

        .. math::

            O = \int_x dx~f(x) \ket{x}\bra{x}

        where :math:`x \in \mathbb{R}`.

        Args:
            basis_states: A set of tensors, in order of the wires, representing position basis states. Each tensor
                has shape ``(*batch_dim)``

        Returns:
            The eigenvalue for each basis state sample, with shape ``(*batch_dim)``
        """
        raise NotImplementedError(
            "This class does not support obtaining the spectrum in position basis"
        )

    def fock_spectrum(self, *basis_states: TensorLike) -> Sequence[float]:
        r"""Provides a diagonal decomposition of the operator in the Fock basis

        An observable that implements this method guarantees it can be written as

        .. math::

            O = \sum_n f(n) \ket{n}\bra{n}

        where :math:`n \in \mathbb{N}_0`.

        Args:
            basis_states: A set of tensors, in order of the wires, representing Fock basis states. Each tensor
                has shape ``(*batch_dim)``

        Returns:
            The eigenvalue for each basis state sample, with shape ``(*batch_dim)``
        """
        raise NotImplementedError(
            "This class does not support obtaining the spectrum in Fock basis"
        )


class Hybrid:
    r"""Mixin for hybrid CV-DV gates

    This mixin adds functionality to split the wires of the gate by type into
    qumodes and qubits. By using this mixin, it enforces the convention that
    qubits come first, followed by qumodes.

    This mixin is also used in static analysis passes to type-check circuits.
    """

    num_qumodes: int | None = None
    """The number of qumodes the gate acts on"""

    type_signature: Sequence[hqml.sa.WireType] | None = None
    """The ordered type signature of each wire"""

    def wire_types(self) -> dict[WiresLike, hqml.sa.WireType]:
        """Identifies the type of each wire in the gate

        Returns:
            A dict mapping wires to their corresponding types
        """

        if (self.num_qumodes is None) == (self.type_signature is None):
            raise ValueError(
                "Gate is improperly defined. It must specify either num_qumodes or type_signature."
            )

        if len(self.wires) < 2:
            raise ValueError("Expected a hybrid gate acting on at least 2 objects")

        type_signature = self.type_signature
        if self.num_qumodes is not None:
            qubits = len(self.wires) - self.num_qumodes
            type_signature = [hqml.sa.Qubit()] * qubits + [
                hqml.sa.Qumode()
            ] * self.num_qumodes

        return {w: s for w, s in zip(self.wires, type_signature)}


class HybridOperation(Hybrid, Operation):
    r"""Hybrid CV-DV quantum gate"""


class FockRepresentation:
    r"""Mixin for operators that define their representation in the Fock basis"""

    @staticmethod
    def compute_fock_matrix(
        wire_dims: tuple[int, ...],  # pyright: ignore[reportUnusedParameter]
        *params: TensorLike,  # pyright: ignore[reportUnusedParameter]
        **hyperparams: TensorLike,  # pyright: ignore[reportUnusedParameter]
    ) -> TensorLike:
        r"""Computes the Fock matrix representation of the gate

        This method should be implemented by any gate to define its representation in the
        Fock basis. It should return a **dense** matrix of shape ``(d, d)``, where
        :math:`d = \prod_i \text{wire\_dims}_i` is the total dimension of the Hilbert space
        of the gate.

        **Details**:

        The ``wire_dims`` argument provides the dimension of each wire in the
        canonical wire order, the same order as ``self.wires``.

        An example using the ``CR`` gate, which has wire types ``[qubit, qumode]``, using
        dimension 2 for the qubit and a dimension of 3 for the qumode:

        >>> hqml.CR.compute_fock_matrix((2, 3), 0.5)
        array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.9689-0.2474j, 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.8776-0.4794j, 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 1.    -0.j    ,
                0.    -0.j    , 0.    -0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    -0.j    ,
                0.9689+0.2474j, 0.    -0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    -0.j    ,
                0.    -0.j    , 0.8776+0.4794j]])

        If ``parameters`` are tensors of a deep learning framework, the returned matrix
        should also be of the same framework so that it can be used in differentiable
        computations. If the gate is nonparametric, this should return a NumPy ``ndarray``.

        Using the same example as above with a JAX array as a parameter, the resulting matrix
        is a differentiable JAX ``Array``:

        >>> hqml.CR.compute_fock_matrix((2, 3), jnp.array(0.5))
        Array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.9689-0.2474j, 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.8776-0.4794j, 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 1.    -0.j    ,
                0.    -0.j    , 0.    -0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    -0.j    ,
                0.9689+0.2474j, 0.    -0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    -0.j    ,
                0.    -0.j    , 0.8776+0.4794j]], dtype=complex128)

        Args:
            wire_dims: The dimension of each wire in the order of ``self.wires``

            *params: The parameters of the gate

            **hyperparams: The hyperparameters of the gate

        Returns:
            The Fock dense matrix representation of the gate, matching the interface of the
            parameters.

        """
        raise MatrixUndefinedError

    def fock_matrix(
        self, wire_dims: Mapping[Any, int], wire_order: WiresLike | None = None
    ) -> TensorLike:
        r"""Computes the dense Fock matrix representation of the gate

        The matrix returned has shape ``(d, d)`` where ``d`` is the total dimension of the
        Hilbert space, as determined from ``wire_dims``.

        **Details**:

        Because operators are symbolic don't know about device-level truncation, the user must
        provide the ``wire_dims`` argument to specify the dimension of each wire. Qubits
        should be explicitly set to dimension ``2``.

        An example using the :math:`R` gate with a Fock truncation of 3:

        >>> hqml.R(0.123, wires=0).fock_matrix({0: 3})
        array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.9924-0.1227j, 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.9699-0.2435j]])

        If ``wire_order`` is not provided, it defaults to ``self.wires``. If the order of the
        wires in ``wire_order`` differs from ``self.wires``, the method automatically
        expands the matrix to the full Hilbert space and permutes the wires as necessary to
        match the requested order.

        For example, defining the :math:`R` gate to act on a composite qubit-qumode system
        with the qubit as wire 0 and the qumode as wire 1, the matrix will be expanded
        as :math:`I_2 \otimes R`:

        >>> hqml.R(0.123, wires=1).fock_matrix({0: 2, 1: 3}, wire_order=(0, 1))
        array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.9924-0.1227j, 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.9699-0.2435j, 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 1.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
                0.9924-0.1227j, 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.9699-0.2435j]])

        Passing a different ``wire_order`` returns a permuted matrix :math:`R \otimes I_2`:

        >>> hqml.R(0.123, wires=1).fock_matrix({0: 2, 1: 3}, wire_order=(1, 0))
        array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 1.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.9924-0.1227j, 0.    +0.j    ,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.9924-0.1227j,
                0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
                0.9699-0.2435j, 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
                0.    +0.j    , 0.9699-0.2435j]])

        If the gate is nonparametric, the returned matrix will be a NumPy ``ndarray``, but if
        the gate has parameters and they are tensors of a deep learning framework, the
        returned matrix will be of the same framework. This function is compatible with
        automatic differentiation and ``@jax.jit`` compilation provided the underlying
        ``compute_fock_matrix`` implementation is as well.

        An example using JAX:

        >>> hqml.R(jnp.array(0.123), wires=0).fock_matrix({0: 3})
        Array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
               [0.    +0.j    , 0.9924-0.1227j, 0.    +0.j    ],
               [0.    +0.j    , 0.    +0.j    , 0.9699-0.2435j]],      dtype=complex128, weak_type=True)

        Developers of new gates should prefer to implement ``compute_fock_matrix``.
        """
        canonical_wire_dims = tuple(wire_dims[w] for w in self.wires)
        matrix = self.compute_fock_matrix(
            canonical_wire_dims, *self.parameters, **self.hyperparameters
        )
        if wire_order is None or self.wires == Wires(wire_order):
            return matrix

        return hqml.math.expand_matrix(
            matrix, self.wires, wire_order=wire_order, wire_dims=wire_dims
        )
