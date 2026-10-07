# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: D102, D107
r"""CV-DV state transfer templates (quantum AD/DA conversion via non-Abelian QSP)

The protocol is due to J. Hastrup, K. Park, J. B. Brask, R. Filip and
U. L. Andersen, "Universal unitary transfer of continuous-variable quantum
states into a few qubits", Phys. Rev. Lett. 128, 110503 (2022),
doi:10.1103/PhysRevLett.128.110503.

These templates follow its presentation as non-Abelian QSP in Sec. IV-B of
Y. Liu, J. M. Martyn, J. Sinanan-Singh, K. C. Smith, S. M. Girvin and
I. L. Chuang, "Toward Mixed Analog-Digital Quantum Signal Processing: Quantum
AD/DA Conversion and the Fourier Transform", IEEE Trans. Signal Process.,
vol. 73, 2025, doi:10.1109/TSP.2025.3599462, whose spacing parameter maps to
these templates as ``Delta = 2 * lmbda``.
"""

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pennylane as qml
from pennylane.decomposition.resources import adjoint_resource_rep
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

import hybridlane as hqml

from ..ops import ConditionalDisplacement, Hybrid

# ---------------------------------------------------------------------------
# Gate sequences
#
# Both directions are built here, once, and reused by the class-level
# ``compute_decomposition`` and by the graph-decomposition rule registered with
# ``qml.add_decomps``.  Keeping a single source of truth stops the two copies
# from drifting apart -- a real risk given the inverse differs from the forward
# sequence only by gate order and a pi shift on each CD phase.
# ---------------------------------------------------------------------------


def _protocol_params(lmbda: float, j: int) -> tuple[float, float]:
    """Displacement magnitudes (a_j, b_j) for the V_j and W_j gates."""
    a_vj = np.pi / (2 ** (j + 1) * lmbda * np.sqrt(2))
    b_wj = lmbda * 2 ** (j - 1) / np.sqrt(2)
    return a_vj, b_wj


def _basis_transform_ops(qubit_wires: list) -> list[Operation]:
    """Final basis transformation B of the forward protocol."""
    n = len(qubit_wires)
    ops = []
    for i, qb in enumerate(qubit_wires):
        ops.append(qml.H(qb))
        if i == n - 1:  # MSB
            ops.append(qml.X(qb))
            ops.append(qml.Z(qb))
        elif i == 0:  # LSB
            ops.append(qml.Z(qb))
        else:  # middle qubits
            ops.append(qml.X(qb))
    return ops


def _cv_to_dv_ops(n: int, lmbda: float, wires: WiresLike) -> list[Operation]:
    r"""Forward protocol: alternating V_j, W_j gates then the basis transform.

    V_j is a sigma_y controlled position displacement, W_j a sigma_x controlled
    momentum displacement.  Ops are built outside any recording context so the
    caller decides whether to queue them.
    """
    all_wires = Wires(wires)
    with qml.QueuingManager.stop_recording():
        qubit_wires = list(all_wires[:n])
        m = all_wires[n]  # qumode wire
        ops = []

        for j in range(1, n + 1):
            qb = qubit_wires[n - j]  # reversed index matching c2qa convention
            a_vj, b_wj = _protocol_params(lmbda, j)

            # V_j: S^dag -> H -> CD(a, pi/2) -> H -> S
            ops.append(qml.adjoint(qml.S)(qb))
            ops.append(qml.H(qb))
            ops.append(ConditionalDisplacement(a_vj, np.pi / 2, [qb, m]))
            ops.append(qml.H(qb))
            ops.append(qml.S(qb))

            # W_j: H -> CD(b, phi) -> H, positive real on the last step only
            phi_wj = 0.0 if j == n else np.pi
            ops.append(qml.H(qb))
            ops.append(ConditionalDisplacement(b_wj, phi_wj, [qb, m]))
            ops.append(qml.H(qb))

        ops.extend(_basis_transform_ops(qubit_wires))

    return ops


def _dv_to_cv_ops(n: int, lmbda: float, wires: WiresLike) -> list[Operation]:
    r"""Adjoint protocol: reverse basis transform, then W_j^dag, V_j^dag.

    Gate order is reversed relative to :func:`_cv_to_dv_ops` and every CD phase
    is shifted by pi, since CD(a, phi)^dag = CD(a, phi + pi).
    """
    all_wires = Wires(wires)
    with qml.QueuingManager.stop_recording():
        qubit_wires = list(all_wires[:n])
        m = all_wires[n]  # qumode wire
        ops = []

        # Reverse basis transformation: forward was H then [X,Z | Z | X],
        # so the adjoint is [Z,X | Z | X] then H.
        for i, qb in enumerate(qubit_wires):
            if i == n - 1:  # MSB
                ops.append(qml.Z(qb))
                ops.append(qml.X(qb))
            elif i == 0:  # LSB
                ops.append(qml.Z(qb))
            else:  # middle qubits
                ops.append(qml.X(qb))
            ops.append(qml.H(qb))

        for j in range(n, 0, -1):
            qb = qubit_wires[n - j]
            a_vj, b_wj = _protocol_params(lmbda, j)

            # W_j^dag: forward phase 0 -> pi, forward pi -> 2pi = 0
            phi_wj_adj = np.pi if j == n else 0.0
            ops.append(qml.H(qb))
            ops.append(ConditionalDisplacement(b_wj, phi_wj_adj, [qb, m]))
            ops.append(qml.H(qb))

            # V_j^dag: forward phase pi/2 -> 3pi/2
            ops.append(qml.adjoint(qml.S)(qb))
            ops.append(qml.H(qb))
            ops.append(ConditionalDisplacement(a_vj, 3 * np.pi / 2, [qb, m]))
            ops.append(qml.H(qb))
            ops.append(qml.S(qb))

    return ops


def _unpack(params: Sequence[TensorLike]) -> tuple[int, float]:
    """Split the (n_qubits, lmbda) parameter pair into plain Python scalars."""
    n_qubits_param, lmbda = params
    return int(n_qubits_param), float(lmbda)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# CV-to-DV state transfer
# ---------------------------------------------------------------------------

_PROTOCOL_RESOURCES = {
    qml.H: 1,
    qml.S: 1,
    adjoint_resource_rep(qml.S): 1,
    ConditionalDisplacement: 1,
    qml.X: 1,
    qml.Z: 1,
}


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
    * Number of parameters: 2

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
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=64)

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
    """

    num_wires = None  # variable: n_qubits + 1 qumode
    num_params = 2
    grad_method = None
    resource_keys: ClassVar = set()

    def __init__(
        self,
        n_qubits: int,
        lmbda: TensorLike = 0.29,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(n_qubits, lmbda, wires=wires, id=id)

    @property
    def type_signature(self):
        n = int(self.parameters[0])  # ty: ignore[invalid-argument-type]
        return tuple([hqml.wires.Qubit()] * n + [hqml.wires.Qumode()])

    @staticmethod
    def compute_decomposition(  # ty: ignore[invalid-method-override]
        *params: TensorLike,
        wires: WiresLike = None,
        **_: dict[str, Any],
    ) -> Sequence[Operation]:
        n, lmbda = _unpack(params)
        return _cv_to_dv_ops(n, lmbda, wires)

    @property
    def resource_params(self):
        return {}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals,
            base_label=base_label or "CV→DV",
            cache=cache,
        )


@qml.register_resources(_PROTOCOL_RESOURCES)
def _state_transfer_cv_to_dv_decomp(*params, wires, **_):
    n, lmbda = _unpack(params)
    for op in _cv_to_dv_ops(n, lmbda, wires):
        qml.apply(op)


qml.add_decomps(StateTransferCVtoDV, _state_transfer_cv_to_dv_decomp)


# ---------------------------------------------------------------------------
# DV-to-CV state transfer (inverse of CV-to-DV)
# ---------------------------------------------------------------------------


class StateTransferDVtoCV(Operation, Hybrid):
    r"""DV-to-CV state transfer using the non-abelian protocol.

    Transfers an n-qubit state to a qumode. This is the inverse (adjoint) of
    :class:`StateTransferCVtoDV`.

    The decomposition applies the reverse basis transformation followed by
    the adjoint V_j and W_j gates in reverse order:

    .. math::

        U_{D/A} = \prod_{j=1}^{n} V_j^\dagger W_j^\dagger \cdot B^\dagger

    where :math:`B` is the basis transformation and :math:`V_j^\dagger`,
    :math:`W_j^\dagger` are the adjoints of the forward gates (CD phase
    shifted by :math:`\pi`).

    .. important::

        **Start the qumode in a squeezed vacuum, not the vacuum.**  The
        protocol maps :math:`|\psi\rangle_Q |0,\Delta\rangle^{\rm sinc}_O
        \mapsto |0\cdots 0\rangle_Q |\psi\rangle^{\rm sinc}_O`: the qubit
        register is returned to :math:`|0\cdots 0\rangle` and the qumode ends
        up holding the state, with no measurement and no post-selection.  That
        only works if the qumode begins in the sinc state
        :math:`|0,\Delta\rangle^{\rm sinc} \propto \int dq\,
        {\rm sinc}(\pi q/\Delta)\,|q\rangle`, which has infinite energy and in
        practice is approximated by a squeezed vacuum
        (:class:`~hybridlane.ops.Squeezing`) of width
        :math:`\sigma_x = e^{-r}/\sqrt{2}`.

        Starting from the vacuum instead leaves the qumode entangled with the
        qubits and the transfer incomplete.

    The ``wires`` attribute is ``(q0, q1, ..., q_{n-1}, qumode)``, with ``q0``
    the most significant bit of the position index
    :math:`q_s = \lambda(2s - (2^n - 1))`.

    **Details**:

    * Number of wires: variable (n_qubits + 1 qumode)
    * Wire arguments: ``[qubit_0, qubit_1, ..., qubit_{n-1}, qumode]``
    * Number of parameters: 2

    Args:
        n_qubits: Number of qubits for the DV register.
        lmbda: Coupling strength parameter (default 0.29).
        wires: Wire labels for the qubits and qumode.
        id: Custom label for the gate.

    **Example**

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
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=256)

        @qp.qnode(dev)
        def circuit(amps):
            qp.StatePrep(amps, wires=qubits)          # state to transfer
            hl.Squeezing(r, 0.0, mode)                # sinc-state stand-in
            hl.StateTransferDVtoCV(n, delta / 2, wires=qubits + [mode])
            return hl.density_matrix(wires=[mode])

        rho = circuit(np.ones(2**n) / np.sqrt(2**n))  # |+++>

    ``rho`` is the transferred qumode state; the qubits are left in
    :math:`|0\cdots0\rangle`.  Check the transfer worked by confirming that
    ``rho`` is close to pure and that the register really did return to
    :math:`|0\cdots0\rangle`; at this cutoff both exceed 0.95, and raising
    ``max_fock_level`` to 512 takes them past 0.98.
    """

    num_wires = None
    num_params = 2
    grad_method = None
    resource_keys: ClassVar = set()

    def __init__(
        self,
        n_qubits: int,
        lmbda: TensorLike = 0.29,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(n_qubits, lmbda, wires=wires, id=id)

    @property
    def type_signature(self):
        n = int(self.parameters[0])  # ty: ignore[invalid-argument-type]
        return tuple([hqml.wires.Qubit()] * n + [hqml.wires.Qumode()])

    @staticmethod
    def compute_decomposition(  # ty: ignore[invalid-method-override]
        *params: TensorLike,
        wires: WiresLike = None,
        **_: dict[str, Any],
    ) -> Sequence[Operation]:
        n, lmbda = _unpack(params)
        return _dv_to_cv_ops(n, lmbda, wires)

    @property
    def resource_params(self):
        return {}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals,
            base_label=base_label or "DV→CV",
            cache=cache,
        )


@qml.register_resources(_PROTOCOL_RESOURCES)
def _state_transfer_dv_to_cv_decomp(*params, wires, **_):
    n, lmbda = _unpack(params)
    for op in _dv_to_cv_ops(n, lmbda, wires):
        qml.apply(op)


qml.add_decomps(StateTransferDVtoCV, _state_transfer_dv_to_cv_decomp)
