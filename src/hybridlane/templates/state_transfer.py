# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: D102, D107
r"""CV-DV state transfer template (quantum AD/DA conversion via non-Abelian QSP)

The protocol is due to J. Hastrup, K. Park, J. B. Brask, R. Filip and
U. L. Andersen, "Universal unitary transfer of continuous-variable quantum
states into a few qubits", Phys. Rev. Lett. 128, 110503 (2022),
doi:10.1103/PhysRevLett.128.110503.

This template follows its presentation as non-Abelian QSP in Sec. IV-B of
Y. Liu, J. M. Martyn, J. Sinanan-Singh, K. C. Smith, S. M. Girvin and
I. L. Chuang, "Toward Mixed Analog-Digital Quantum Signal Processing: Quantum
AD/DA Conversion and the Fourier Transform", IEEE Trans. Signal Process.,
vol. 73, 2025, doi:10.1109/TSP.2025.3599462, whose spacing parameter maps to
this template as ``Delta = 2 * lmbda``.

The A/D direction is :class:`StateTransferCVtoDV`; the D/A direction is its
adjoint, ``qp.adjoint(StateTransferCVtoDV(...))``.
"""

from typing import ClassVar

import numpy as np
import pennylane as qp
from pennylane.decomposition.resources import adjoint_resource_rep
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

import hybridlane as hl

from ..ops import ConditionalDisplacement, Hybrid


def _protocol_params(lmbda: float, j: int) -> tuple[float, float]:
    """Displacement magnitudes (a_j, b_j) for the V_j and W_j gates."""
    a_vj = np.pi / (2 ** (j + 1) * lmbda * np.sqrt(2))
    b_wj = lmbda * 2 ** (j - 1) / np.sqrt(2)
    return a_vj, b_wj


def _basis_transform_ops(qubit_wires: list) -> list[Operation]:
    """Final basis transformation B of the protocol."""
    n = len(qubit_wires)
    ops = []
    for i, qb in enumerate(qubit_wires):
        ops.append(qp.H(qb))
        if i == n - 1:  # MSB
            ops.append(qp.X(qb))
            ops.append(qp.Z(qb))
        elif i == 0:  # LSB
            ops.append(qp.Z(qb))
        else:  # middle qubits
            ops.append(qp.X(qb))
    return ops


def _cv_to_dv_ops(n: int, lmbda: float, wires: WiresLike) -> list[Operation]:
    r"""Alternating V_j, W_j gates then the basis transform.

    V_j is a sigma_y controlled position displacement, W_j a sigma_x controlled
    momentum displacement.  Ops are built outside any recording context so the
    caller decides whether to queue them.
    """
    all_wires = Wires(wires)
    with qp.QueuingManager.stop_recording():
        qubit_wires = list(all_wires[:n])
        m = all_wires[n]  # qumode wire
        ops = []

        for j in range(1, n + 1):
            qb = qubit_wires[n - j]  # reversed index matching c2qa convention
            a_vj, b_wj = _protocol_params(lmbda, j)

            # V_j: S^dag -> H -> CD(a, pi/2) -> H -> S
            ops.append(qp.adjoint(qp.S)(qb))
            ops.append(qp.H(qb))
            ops.append(ConditionalDisplacement(a_vj, np.pi / 2, [qb, m]))
            ops.append(qp.H(qb))
            ops.append(qp.S(qb))

            # W_j: H -> CD(b, phi) -> H, positive real on the last step only
            phi_wj = 0.0 if j == n else np.pi
            ops.append(qp.H(qb))
            ops.append(ConditionalDisplacement(b_wj, phi_wj, [qb, m]))
            ops.append(qp.H(qb))

        ops.extend(_basis_transform_ops(qubit_wires))

    return ops


class StateTransferCVtoDV(Operation, Hybrid):
    r"""CV-to-DV state transfer using the non-abelian protocol.

    Transfers a qumode state to n qubits using alternating V_j and W_j gates
    followed by a basis transformation.

    V_j implements a sigma_y controlled position displacement.

    W_j implements a sigma_x controlled momentum displacement.

    This is the *measurement* direction: with the qubits initialised to
    :math:`|0\cdots 0\rangle` and the qumode holding :math:`|\psi\rangle`,
    measuring the qubit register in the computational basis samples the
    position wavefunction,

    .. math::

        P(s) \approx |\psi(q_s)|^2 \cdot 2\lambda,
        \qquad q_s = \lambda(2s - (2^n - 1)).

    Equivalently, as a coherent transfer it maps
    :math:`|0\cdots 0\rangle_Q |\psi\rangle_O \mapsto |\psi\rangle_Q
    |0,\Delta\rangle^{\rm sinc}_O`, leaving the qumode in the sinc state.  No
    measurement or post-selection is involved in either reading.

    The ``wires`` attribute is ``(q0, q1, ..., q_{n-1}, qumode)``, with ``q0``
    the most significant bit of :math:`s`.

    **Details**:

    * Number of wires: variable (n_qubits + 1 qumode)
    * Wire arguments: ``[qubit_0, qubit_1, ..., qubit_{n-1}, qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    Args:
        n_qubits: Number of qubits for the DV register.
        lmbda: Coupling strength parameter (default 0.29).
        wires: Wire labels for the qubits and qumode.
        id: Custom label for the gate.

    **Example**

    Read out the position wavefunction of a qumode onto four qubits.  The
    operation takes ``n_qubits + 1`` wires, qubits first and the qumode last:

    .. code-block:: python

        import pennylane as qp
        import hybridlane as hl
        from hybridlane.wires import BasisMap, ComputationalBasis

        qp.decomposition.enable_graph()

        n, lmbda = 4, 0.29
        qubits, mode = list(range(n)), n
        dev = qp.device("default.hybrid", fock_level=64)

        @qp.set_shots(4096)
        @qp.qnode(dev)
        def circuit():
            hl.FockState(1, [mode + 1, mode])          # qumode state to read out
            hl.StateTransferCVtoDV(n, lmbda, wires=qubits + [mode])
            schema = BasisMap(
                {qp.wires.Wires(w): ComputationalBasis.Discrete for w in qubits}
            )
            return hl.sample(schema=schema)

    Outcome ``s`` occurs with probability :math:`|\psi(q_s)|^2 \cdot 2\lambda`,
    where :math:`q_s = \lambda(2s - (2^n - 1))`.

    **The D/A direction**

    The digital-to-analog transfer is the adjoint of this template, so use
    ``qp.adjoint`` rather than a separate operation.  It is coherent and
    deterministic -- no measurement and no post-selection -- mapping
    :math:`|\psi\rangle_Q |0,\Delta\rangle^{\rm sinc}_O \mapsto
    |0\cdots 0\rangle_Q |\psi\rangle^{\rm sinc}_O`.

    .. important::

        **Start the qumode in a squeezed vacuum, not the vacuum.**  The adjoint
        only performs the transfer if the qumode begins in the sinc state
        :math:`|0,\Delta\rangle^{\rm sinc} \propto \int dq\,
        {\rm sinc}(\pi q/\Delta)\,|q\rangle`, which has infinite energy and in
        practice is approximated by a squeezed vacuum
        (:class:`~hybridlane.ops.Squeezing`) of width
        :math:`\sigma_x = e^{-r}/\sqrt{2}`.  This operation does not prepare
        that state -- it is a precondition on the input, and starting from the
        vacuum instead leaves the qumode entangled with the qubits and the
        transfer incomplete.

    Move a three-qubit state into a qumode.  Note the
    :class:`~hybridlane.ops.Squeezing` on the qumode *before* the transfer --
    it stands in for the sinc state and is not optional:

    .. code-block:: python

        import numpy as np
        import pennylane as qp
        import hybridlane as hl

        qp.decomposition.enable_graph()

        n, delta, r = 3, 1.2, 2.5           # lmbda = delta / 2
        qubits, mode = list(range(n)), n
        dev = qp.device("default.hybrid", fock_level=256)

        @qp.qnode(dev)
        def circuit(amps):
            qp.StatePrep(amps, wires=qubits)          # state to transfer
            hl.Squeezing(r, 0.0, mode)                # sinc-state stand-in
            qp.adjoint(hl.StateTransferCVtoDV(n, delta / 2, wires=qubits + [mode]))
            return hl.density_matrix(wires=[mode])

        rho = circuit(np.ones(2**n) / np.sqrt(2**n))  # |+++>

    ``rho`` is the transferred qumode state; the qubits are left in
    :math:`|0\cdots0\rangle`.  Check the transfer worked by confirming that
    ``rho`` is close to pure and that the register really did return to
    :math:`|0\cdots0\rangle`; at this cutoff both exceed 0.95, and raising
    ``fock_level`` to 512 takes them past 0.98.
    """

    num_wires = None  # variable: n_qubits + 1 qumode
    num_params = 1  # pyright: ignore[reportIncompatibleMethodOverride]
    ndim_params = (0,)  # pyright: ignore[reportIncompatibleMethodOverride]
    grad_method = None  # pyright: ignore[reportIncompatibleMethodOverride]

    resource_keys: ClassVar = {"n_qubits"}

    def __init__(
        self,
        n_qubits: int,
        lmbda: TensorLike = 0.29,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(lmbda, wires=wires, id=id)
        self.hyperparameters["n_qubits"] = int(n_qubits)

    @property
    def type_signature(self):
        n = self.hyperparameters["n_qubits"]
        return tuple([hl.wires.Qubit()] * n + [hl.wires.Qumode()])

    @property
    def resource_params(self):
        return {"n_qubits": self.hyperparameters["n_qubits"]}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals,
            base_label=base_label or "CV→DV",
            cache=cache,
        )


def _state_transfer_resources(n_qubits):
    """Gate counts for an ``n_qubits`` transfer.

    Each of the n (V_j, W_j) pairs costs 4 H, one S, one S^dag and two CDs; the
    basis transform adds one H per qubit plus the X/Z pattern of
    :func:`_basis_transform_ops`, which degenerates to a single X and Z at
    ``n = 1`` where the one qubit is both the LSB and the MSB.
    """
    n = n_qubits
    return {
        qp.H: 5 * n,
        qp.S: n,
        adjoint_resource_rep(qp.S): n,
        ConditionalDisplacement: 2 * n,
        qp.X: 1 if n == 1 else n - 1,
        qp.Z: 1 if n == 1 else 2,
    }


@qp.register_resources(_state_transfer_resources)
def _state_transfer_cv_to_dv_decomp(lmbda, wires, n_qubits, **_):
    for op in _cv_to_dv_ops(n_qubits, float(lmbda), wires):
        qp.apply(op)


qp.add_decomps(StateTransferCVtoDV, _state_transfer_cv_to_dv_decomp)
