# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

import itertools

import pennylane as qp
import pennylane.numpy as np
import pytest
from scipy.special import gammaln

import hybridlane as hl
from hybridlane.templates.non_abelian_qsp import GKPState, SqueezedCatState


# The distribution p(n) for cat states
def cat_state_probs(alpha, ns, odd: bool):
    alpha_sq = np.abs(alpha) ** 2  # ty:ignore[unresolved-attribute]
    log_factorial = gammaln(ns + 1)
    log_P_coh = -alpha_sq + (ns * np.log(alpha_sq)) - log_factorial  # ty:ignore[unresolved-attribute]
    log_norm = np.log(2) - np.log(1 + np.exp(-2 * alpha_sq))  # ty:ignore[unresolved-attribute]
    log_Pn = log_norm + log_P_coh
    log_Pn[ns % 2 != int(odd)] = -np.inf  # ty:ignore[unresolved-attribute]
    return np.exp(log_Pn)  # ty:ignore[unresolved-attribute]


class TestSqueezedCatState:
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "alpha,parity", list(itertools.product([3, 4, 5, 6], ("even", "odd")))
    )
    def test_parity(self, alpha, parity):
        fock_level = 128
        dev = qp.device("default.hybrid", fock_level=fock_level)

        @qp.qnode(dev)
        def circuit(alpha):
            SqueezedCatState(alpha, np.pi / 2, parity=parity, wires=["q", "m"])  # ty:ignore[unresolved-attribute]

            qp.H("a")
            hl.ConditionalParity(["a", "m"])
            qp.H("a")
            return hl.expval(qp.Z("a"))

        parity_expval = circuit(alpha)
        if parity == "even":
            assert parity_expval >= 0.95
        else:
            assert parity_expval <= -0.95

    @pytest.mark.integration
    @pytest.mark.parametrize("alpha,parity", [(3, "even"), (6, "odd")])
    def test_qubit_fidelity_and_mean_photon_count(self, alpha, parity):
        dev = qp.device("default.hybrid", fock_level=128)

        @qp.qnode(dev)
        def circuit(alpha):
            SqueezedCatState(alpha, np.pi / 2, parity=parity, wires=["q", "m"])  # ty:ignore[unresolved-attribute]
            return hl.expval(qp.Z("q")), hl.expval(hl.N("m"))

        expval_z, expval_n = circuit(alpha)
        fidelity = (1 + expval_z) / 2
        expected = 1 - np.pi**2 / (64 * alpha**2)  # eq. 393  # ty:ignore[unresolved-attribute]
        assert fidelity >= expected

        n = np.arange(256)  # ty:ignore[unresolved-attribute]
        p_n = cat_state_probs(alpha, n, parity == "odd")
        expected_mean = (n * p_n).sum()
        assert np.allclose(expval_n, expected_mean, rtol=1e-2)  # ty:ignore[unresolved-attribute]


class TestGKPState:
    @pytest.mark.parametrize(
        "delta,expected_repetitions", [(0.4, 1), (0.3, 3), (0.2, 7), (0.1, 31)]
    )
    @pytest.mark.unit
    def test_repetitions(self, delta, expected_repetitions):
        op = hl.GKPState(delta, logical_state=0, wires=["q", "m"])
        assert op.hyperparameters["repetitions"] == expected_repetitions

    @pytest.mark.integration
    @pytest.mark.parametrize("codeword", (0, 1))
    def test_stabilizer(self, codeword):
        fock_level = 128
        dev = qp.device("default.hybrid", fock_level=fock_level)

        @qp.qnode(dev)
        def circuit(codeword, delta=1):
            GKPState(delta, logical_state=codeword, wires=["q", "m"])

            # SBS stabilizer measurement, fig. 9
            alpha = np.sqrt(np.pi / 8)  # ty:ignore[unresolved-attribute]
            lam = -alpha * delta**2
            hl.YCD(lam, 0, wires=["q", "m"])
            hl.XCD(2 * alpha, np.pi / 2, wires=["q", "m"])  # ty:ignore[unresolved-attribute]
            hl.YCD(lam, 0, wires=["q", "m"])
            return hl.expval(qp.Z("q"))

        stabilizer_expval = circuit(codeword, 0.34)
        assert stabilizer_expval >= 0.99  # from table 3

    @pytest.mark.integration
    @pytest.mark.parametrize("codeword", (0, 1))
    def test_error(self, codeword):
        fock_level = 256
        dev = qp.device("default.hybrid", fock_level=fock_level)

        @qp.qnode(dev)
        def circuit(codeword, delta=1):
            GKPState(delta, logical_state=codeword, wires=["q", "m"])
            return hl.expval(qp.Z("q"))

        errors_dict = qp.resource.algo_error(circuit)(codeword, 0.34)
        assert errors_dict["SpectralNormError"].error > 0  # ty:ignore[invalid-argument-type]
