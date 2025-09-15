# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from typing import Iterable, Sequence
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ..sa.base import ComputationalBasis


class Spectral:
    r"""Mixin for observables that have an infinite number of eigenvalues (spectrum)

    Instead of enumerating all eigenvalues like normal Pennylane observables (because that is no
    longer possible), this mixin provides a general framework for observables to define their spectrum,
    a function :math:`f: \mathcal{B} \rightarrow \mathbb{R}` from basis states to eigenvalues.
    """

    @property
    def natural_basis(self) -> ComputationalBasis:
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

    num_qumodes: int
    """The number of qumodes the gate acts on"""

    wires: Wires

    def split_wires(self) -> tuple[Wires, Wires]:
        """Splits the wires into qubits and qumodes

        Returns:
            qubits: The wires representing the qubits this operator acts on

            qumodes: The wires representing the qumodes this operator acts on
        """

        if not isinstance(self.wires, Iterable):
            raise ValueError("Expected a hybrid gate acting on at least 2 objects")

        wires = Wires(self.wires)
        qubits, qumodes = wires[: -self.num_qumodes], wires[-self.num_qumodes :]
        return qubits, qumodes
