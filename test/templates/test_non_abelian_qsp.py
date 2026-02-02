# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import itertools

import numpy as np
import pennylane as qml
import pytest
from scipy.special import gammaln

import hybridlane as hqml
from hybridlane.templates.non_abelian_qsp import SqueezedCatState


# The distribution p(n) for even parity cat states
def cat_state_probs(alpha, ns, odd: bool):
    alpha_sq = np.abs(alpha) ** 2
    log_factorial = gammaln(ns + 1)
    log_P_coh = -alpha_sq + (ns * np.log(alpha_sq)) - log_factorial
    log_norm = np.log(2) - np.log(1 + np.exp(-2 * alpha_sq))
    log_Pn = log_norm + log_P_coh
    log_Pn[ns % 2 != int(odd)] = -np.inf
    return np.exp(log_Pn)


@pytest.mark.slow
class TestSqueezedCatState:
    @pytest.mark.parametrize(
        "alpha,parity", itertools.product([3, 4, 5, 6], ("even", "odd"))
    )
    def test_parity(self, alpha, parity):
        fock_level = 256
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=fock_level)

        @qml.qnode(dev)
        def circuit(alpha):
            SqueezedCatState(alpha, 0, parity=parity, wires=["q", "m"])

            qml.H("a")
            hqml.ConditionalParity(["a", "m"])
            qml.H("a")
            return hqml.expval(qml.Z("a"))

        parity_expval = circuit(alpha)
        if parity == "even":
            assert parity_expval >= 0.95
        else:
            assert parity_expval <= -0.95

    @pytest.mark.parametrize("alpha,parity", [(3, "even"), (6, "odd")])
    def test_qubit_fidelity_and_mean_photon_count(self, alpha, parity):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=256)

        @qml.qnode(dev)
        def circuit(alpha):
            SqueezedCatState(alpha, 0, parity=parity, wires=["q", "m"])
            return hqml.expval(qml.Z("q")), hqml.expval(hqml.N("m"))

        expval_z, expval_n = circuit(alpha)
        fidelity = (1 + expval_z) / 2
        expected = 1 - np.pi**2 / (64 * alpha**2)  # eq. 393
        assert fidelity >= expected

        n = np.arange(256)
        p_n = cat_state_probs(alpha, n, parity == "odd")
        expected_mean = (n * p_n).sum()
        assert np.allclose(expval_n, expected_mean, rtol=1e-2)
