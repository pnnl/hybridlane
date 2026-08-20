# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Tests for StateTransferCVtoDV and StateTransferDVtoCV.

CV->DV samples the qumode position wavefunction on the qubit register,
P(s) ~ |psi(q_s)|^2 * 2*lmbda with q_s = lmbda(2s - (2^n - 1)).

DV->CV is the adjoint and returns the register to |0...0> while the qumode
takes up the state, provided the qumode starts squeezed. Success is measured
by the population of |0...0> and the purity of the qumode.

References:
    J. Hastrup, K. Park, J. B. Brask, R. Filip and U. L. Andersen, "Universal
    unitary transfer of continuous-variable quantum states into a few qubits",
    Phys. Rev. Lett. 128, 110503 (2022), doi:10.1103/PhysRevLett.128.110503.

    Y. Liu, J. M. Martyn, J. Sinanan-Singh, K. C. Smith, S. M. Girvin and
    I. L. Chuang, "Toward Mixed Analog-Digital Quantum Signal Processing:
    Quantum AD/DA Conversion and the Fourier Transform", IEEE Trans. Signal
    Process. 73 (2025), doi:10.1109/TSP.2025.3599462, Sec. IV-B.
"""

from collections import Counter

import numpy as np
import pennylane as qp
import pytest
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane.ops import ConditionalDisplacement
from hybridlane.templates import StateTransferCVtoDV, StateTransferDVtoCV
from hybridlane.templates.non_abelian_qsp import SqueezedCatState
from hybridlane.wires import BasisMap, ComputationalBasis


def sampling_grid(n_qubits, lmbda):
    s = np.arange(2**n_qubits)
    return lmbda * (2 * s - (2**n_qubits - 1))


# Classical fidelity (sum_s sqrt(p_sim * p_ideal))^2
def fidelity(p_sim, p_ideal):
    return float(np.sum(np.sqrt(np.clip(p_sim, 0, None) * np.clip(p_ideal, 0, None))) ** 2)


def normalized(psi):
    p = np.abs(psi) ** 2
    return p / p.sum()


# Sample counts -> probabilities over bitstrings, wire 0 the most significant bit
def sample_probs(result, qubit_keys, shots):
    n = len(qubit_keys)
    strings = ["".join(str(result.data[k][i]) for k in qubit_keys) for i in range(shots)]
    counts = Counter(strings)
    return np.array([counts.get(format(i, f"0{n}b"), 0) / shots for i in range(2**n)])


# Run DV->CV and report (population of |0...0>, purity of the qumode)
def transfer_diagnostics(n, lmbda, squeezing, fock, amps):
    qubits = list(range(n))
    dev = qp.device("bosonicqiskit.hybrid", max_fock_level=fock)

    @qp.qnode(dev)
    def circuit(a):
        qp.StatePrep(a, wires=qubits)
        if squeezing:
            hl.Squeezing(squeezing, 0.0, n)
        StateTransferDVtoCV(n, lmbda, wires=[*qubits, n])
        return hl.state()

    a = np.asarray(amps, dtype=complex)
    vec = np.asarray(circuit(a / np.linalg.norm(a))).reshape((2,) * n + (fock,))

    zeros = vec[(0,) * n]
    flat = vec.reshape(-1, fock)
    rho = flat.conj().T @ flat
    rho = rho / np.trace(rho)
    return float(np.real(np.vdot(zeros, zeros))), float(np.real(np.trace(rho @ rho)))


@pytest.mark.usefixtures("enable_graph_decomp")
class TestStateTransferCVtoDV:
    @pytest.mark.unit
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_resource_params(self, n):
        op = StateTransferCVtoDV(n, 0.29, wires=[f"q{i}" for i in range(n)] + ["m"])
        assert op.resource_params == {}

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_type_signature(self, n):
        op = StateTransferCVtoDV(n, 0.29, wires=[f"q{i}" for i in range(n)] + ["m"])
        sig = op.type_signature
        assert len(sig) == n + 1
        assert all(isinstance(s, hl.wires.Qubit) for s in sig[:n])
        assert isinstance(sig[n], hl.wires.Qumode)

    @pytest.mark.unit
    def test_labels(self):
        wires = ["q0", "q1", "q2", "q3", "m"]
        assert StateTransferCVtoDV(4, 0.29, wires=wires).label() == "CV→DV"
        assert StateTransferDVtoCV(4, 0.29, wires=wires).label() == "DV→CV"

    # One CD per V_j and per W_j, j = 1..n
    @pytest.mark.unit
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_cd_gate_count(self, n):
        ops = StateTransferCVtoDV.compute_decomposition(
            n, 0.29, wires=[f"q{i}" for i in range(n)] + ["m"]
        )
        assert sum(isinstance(o, ConditionalDisplacement) for o in ops) == 2 * n

    @pytest.mark.integration
    @pytest.mark.bq
    @pytest.mark.parametrize(
        ("name", "psi", "threshold"),
        [
            ("vacuum", lambda q: np.pi ** (-0.25) * np.exp(-(q**2) / 2), 0.95),
            ("fock1", lambda q: np.pi ** (-0.25) * np.sqrt(2) * q * np.exp(-(q**2) / 2), 0.95),
        ],
    )
    def test_readout_fidelity(self, name, psi, threshold):
        n, lmbda, shots = 4, 0.29, 4096
        qubits = [f"q{i}" for i in range(n)]
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=64)

        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circuit():
            if name == "fock1":
                hl.FockState(1, ["anc", "m"])
            else:
                hl.Displacement(0.0, 0.0, "m")
            StateTransferCVtoDV(n, lmbda, wires=[*qubits, "m"])
            schema = BasisMap({Wires(w): ComputationalBasis.Discrete for w in qubits})
            return hl.sample(schema=schema)

        q_s = sampling_grid(n, lmbda)
        fid = fidelity(sample_probs(circuit(), qubits, shots), normalized(psi(q_s)))
        assert fid >= threshold, f"{name} fidelity F={fid:.4f} below {threshold}"

    # State prep and transfer are both approximate, so the threshold is looser
    @pytest.mark.integration
    @pytest.mark.bq
    @pytest.mark.slow
    def test_readout_fidelity_cat(self):
        alpha, n, lmbda, shots = 2, 4, 0.29, 4096
        qubits = [f"q{i}" for i in range(n)]
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=128)

        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circuit():
            SqueezedCatState(alpha, np.pi / 2, delta=1, parity="even", wires=["cat_anc", "m"])
            StateTransferCVtoDV(n, lmbda, wires=[*qubits, "m"])
            schema = BasisMap({Wires(w): ComputationalBasis.Discrete for w in qubits})
            return hl.sample(schema=schema)

        q_s = sampling_grid(n, lmbda)
        ideal = normalized(
            np.exp(-((q_s - np.sqrt(2) * alpha) ** 2) / 2)
            + np.exp(-((q_s + np.sqrt(2) * alpha) ** 2) / 2)
        )
        fid = fidelity(sample_probs(circuit(), qubits, shots), ideal)
        assert fid >= 0.90, f"cat fidelity F={fid:.4f} below 0.90"


@pytest.mark.usefixtures("enable_graph_decomp")
class TestStateTransferDVtoCV:
    """hl.state only round-trips integer wire labels 0..k-1 on Bosonic Qiskit,
    so the qubits are 0..n-1 and the qumode is n.
    """

    N = 4
    LMBDA = 0.29
    FOCK = 64

    DELTA = np.sqrt(2)  # Liu et al. Fig. 4(b) spacing, = 2 * lmbda
    SQUEEZING = 2.0

    @property
    def _wires(self):
        return [*range(self.N), self.N]

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_resource_params(self, n):
        op = StateTransferDVtoCV(n, 0.29, wires=[f"q{i}" for i in range(n)] + ["m"])
        assert op.resource_params == {}

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_type_signature(self, n):
        op = StateTransferDVtoCV(n, 0.29, wires=[f"q{i}" for i in range(n)] + ["m"])
        sig = op.type_signature
        assert len(sig) == n + 1
        assert all(isinstance(s, hl.wires.Qubit) for s in sig[:n])
        assert isinstance(sig[n], hl.wires.Qumode)

    # Inverting reorders gates but must not change the hardware cost
    @pytest.mark.unit
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_cd_gate_count(self, n):
        ops = StateTransferDVtoCV.compute_decomposition(
            n, 0.29, wires=[f"q{i}" for i in range(n)] + ["m"]
        )
        assert sum(isinstance(o, ConditionalDisplacement) for o in ops) == 2 * n

    # CD(a, phi)^dag = CD(a, phi + pi)
    @pytest.mark.unit
    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_cd_phases_are_inverted(self, n):
        wires = [f"q{i}" for i in range(n)] + ["m"]
        fwd = StateTransferCVtoDV.compute_decomposition(n, 0.29, wires=wires)
        inv = StateTransferDVtoCV.compute_decomposition(n, 0.29, wires=wires)
        assert len(fwd) == len(inv)

        fwd_cds = [o for o in fwd if isinstance(o, ConditionalDisplacement)]
        inv_cds = [o for o in reversed(inv) if isinstance(o, ConditionalDisplacement)]
        assert len(fwd_cds) == len(inv_cds) == 2 * n

        for f_op, i_op in zip(fwd_cds, inv_cds, strict=True):
            assert i_op.wires == f_op.wires
            a_f, phi_f = np.asarray(f_op.parameters[:2], dtype=float)
            a_i, phi_i = np.asarray(i_op.parameters[:2], dtype=float)
            assert np.isclose(a_i, a_f)
            offset = (phi_i - phi_f) % (2 * np.pi)
            assert np.isclose(offset, np.pi), f"CD phase offset {offset:.6f}, expected pi"

    # |s> maps onto the same grid the forward direction samples, and wire 0 is
    # the most significant bit
    @pytest.mark.integration
    @pytest.mark.bq
    def test_mean_position_of_basis_states(self):
        n, lmbda = self.N, self.LMBDA
        qubits = list(range(n))
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=self.FOCK)

        @qp.qnode(dev)
        def circuit(bits):
            for w, b in zip(qubits, bits, strict=True):
                if b:
                    qp.X(w)
            StateTransferDVtoCV(n, lmbda, wires=self._wires)
            return hl.expval(hl.QuadX(n))

        q_s = sampling_grid(n, lmbda)
        for s in range(2**n):
            x = float(np.real(circuit([int(b) for b in format(s, f"0{n}b")])))
            assert np.isclose(x, q_s[s], atol=1e-6), f"s={s}: <x>={x:.6f}, want {q_s[s]:.6f}"

    # The transfer is deterministic, but only from a squeezed qumode; from the
    # vacuum a third of the amplitude never returns to |0...0>
    @pytest.mark.integration
    @pytest.mark.bq
    @pytest.mark.slow
    def test_transfer_is_unentangling(self):
        n, lmbda, fock = 3, self.DELTA / 2, 256
        amps = np.ones(2**n)

        p0, purity = transfer_diagnostics(n, lmbda, self.SQUEEZING, fock, amps)
        assert p0 >= 0.98, f"register returned to |0...0> with only P={p0:.4f}"
        assert purity >= 0.96, f"qumode left mixed, purity={purity:.4f}"

        p0_vacuum, _ = transfer_diagnostics(n, lmbda, 0.0, fock, amps)
        assert p0_vacuum < 0.8, f"vacuum input unexpectedly worked, P={p0_vacuum:.4f}"

    # Liu et al. Fig. 4(b) reports 0.858 for |+++> at Delta = sqrt(2), a sinc
    # state approximated by a Gaussian of width e^-1.12, and Fock 128
    @pytest.mark.integration
    @pytest.mark.bq
    @pytest.mark.slow
    def test_matches_published_purity(self):
        r = -np.log(np.exp(-1.12) * np.sqrt(2))  # sigma_x = e^-r / sqrt(2)
        _, purity = transfer_diagnostics(3, self.DELTA / 2, r, 128, np.ones(8))
        assert np.isclose(purity, 0.858, atol=0.02), f"|+++> purity {purity:.4f} vs 0.858"

    # Consistency check that the two decompositions are adjoints on device
    @pytest.mark.integration
    @pytest.mark.bq
    def test_roundtrip_is_identity(self):
        n, lmbda, shots = self.N, self.LMBDA, 512
        qubits = list(range(n))
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=self.FOCK)
        schema = BasisMap({Wires(w): ComputationalBasis.Discrete for w in qubits})

        @qp.set_shots(shots)
        @qp.qnode(dev)
        def circuit(bits):
            for w, b in zip(qubits, bits, strict=True):
                if b:
                    qp.X(w)
            StateTransferDVtoCV(n, lmbda, wires=self._wires)
            StateTransferCVtoDV(n, lmbda, wires=self._wires)
            return hl.sample(schema=schema)

        for s in [0, 5, 10, 15]:
            probs = sample_probs(circuit([int(b) for b in format(s, f"0{n}b")]), qubits, shots)
            assert probs[s] >= 0.99, f"s={s}: recovered with P={probs[s]:.4f}"
