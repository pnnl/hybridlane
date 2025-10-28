# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from __future__ import annotations

from collections.abc import Sequence

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

    wires: Wires

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
