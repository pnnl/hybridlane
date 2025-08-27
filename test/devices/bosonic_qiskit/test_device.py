import importlib.util
import sys
from functools import partial
from unittest import mock

import numpy as np
import pennylane as qml
import pytest
import scipy.stats
from pennylane.exceptions import DeviceError

import hybridlane as hqml
from hybridlane.measurements import FockTruncation, Truncation
from hybridlane.sa.exceptions import StaticAnalysisError

from ...util import poisson_test


def test_package_works_without_bosonic_qiskit(monkeypatch):
    """Test that hybrid-circuit can be imported without bosonic qiskit"""

    monkeypatch.delitem(sys.modules, "c2qa", raising=False)

    import hybridlane as hqml
    from hybridlane.devices import BosonicQiskitDevice


missing_bosonic_qiskit = importlib.util.find_spec("c2qa") is None


# Unit tests should go in here
@pytest.mark.skipif(missing_bosonic_qiskit, reason="Requires bosonic qiskit")
class TestBosonicQiskitDevice:
    def test_device_is_registered(self):
        from hybridlane.devices import BosonicQiskitDevice

        dev = qml.device("hybrid.bosonicqiskit")
        assert isinstance(dev, BosonicQiskitDevice)

    def test_non_power_of_two_truncation(self):
        trunc = FockTruncation.all_fock_space([0, 1], {0: 2, 1: 7})
        dev = qml.device("hybrid.bosonicqiskit", truncation=trunc)

        @qml.qnode(dev)
        def circuit():
            hqml.ConditionalDisplacement(1.0, 0, [1, 0])
            return hqml.expval(hqml.NumberOperator(1))

        with pytest.raises(DeviceError):
            circuit()

    def test_unspecified_qumode(self):
        with pytest.raises(DeviceError):
            qml.device("hybrid.bosonicqiskit", wires=[0, 1], max_fock_level=8)

    def test_no_inferrable_truncation(self):
        # This circuit has a qumode that should be detected through static analysis,
        # but no truncation is provided.
        dev = qml.device("hybrid.bosonicqiskit")

        @qml.qnode(dev)
        def circuit():
            qml.Rotation(0.5, 0)
            return hqml.expval(hqml.NumberOperator(0))

        with pytest.raises(DeviceError):
            circuit()

    def test_infer_qubits(self):
        # This circuit should be detected as all qubit and therefore automagically
        # derive a truncation of 2 for each qubit. It'll then fail because
        # we only simulate hybrid programs.
        dev = qml.device("hybrid.bosonicqiskit")

        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            return hqml.expval(qml.Z(0) @ qml.X(1))

        with pytest.raises(DeviceError):
            circuit()

    def test_non_fock_truncation(self):
        trunc = mock.Mock(spec=Truncation)
        dev = qml.device("hybrid.bosonicqiskit", truncation=trunc)

        @qml.qnode(dev)
        def circuit():
            hqml.ConditionalDisplacement(1.0, 0, [1, 0])
            return hqml.expval(hqml.NumberOperator(1))

        with pytest.raises(DeviceError):
            circuit()

    def test_wires_aliased_by_operation(self):
        dev = qml.device(
            "hybrid.bosonicqiskit", wires=[0, 1], qumodes=[1], max_fock_level=8
        )

        @qml.qnode(dev)
        def circuit():
            hqml.ConditionalDisplacement(
                1.0, 0, [1, 0]
            )  # wire 0 established as a qubit here
            return hqml.expval(hqml.NumberOperator(0))  # measure wire 0 in fock basis

        with pytest.raises(StaticAnalysisError):
            circuit()

    @pytest.mark.parametrize(
        "obs",
        (
            hqml.NumberOperator(0) @ qml.PauliZ(0),
            hqml.NumberOperator(0) + qml.PauliZ(0),
            qml.s_prod(0.5, hqml.NumberOperator(0) @ qml.PauliZ(0)),
            qml.s_prod(0.5, hqml.NumberOperator(0) + qml.PauliZ(0)),
        ),
    )
    def test_wires_aliased_by_observable(self, obs):
        dev = qml.device(
            "hybrid.bosonicqiskit", wires=[0, 1], qumodes=[0], max_fock_level=8
        )

        @qml.qnode(dev)
        def circuit():
            return hqml.expval(obs)

        with pytest.raises(StaticAnalysisError):
            circuit()


# Integration circuit-level tests go in here
@pytest.mark.skipif(missing_bosonic_qiskit, reason="Requires bosonic qiskit")
class TestExampleCircuits:
    def test_vacuum_expval(self):
        # The simplest test you could do, checking the vacuum state |0> has <n> = 0

        dev = qml.device(
            "hybrid.bosonicqiskit", wires=[0], qumodes=[0], max_fock_level=16
        )

        @qml.qnode(dev)
        def circuit():
            return hqml.expval(hqml.NumberOperator(0))

        result = circuit()
        assert np.ndim(result) == 0
        assert np.isclose(result, 0)

    def test_vacuum_var(self):
        # Checking the vacuum state |0> has Var(n) = 0 since it's a definite eigenstate
        dev = qml.device(
            "hybrid.bosonicqiskit", wires=[0], qumodes=[0], max_fock_level=16
        )

        @qml.qnode(dev)
        def circuit():
            return hqml.var(hqml.NumberOperator(0))

        result = circuit()
        assert np.ndim(result) == 0
        assert np.isclose(result, 0)

    def test_heisenberg_uncertainty(self):
        dev = qml.device(
            "hybrid.bosonicqiskit",
            wires=[0],
            qumodes=[0],
            max_fock_level=16,
            hbar=2,
        )

        @qml.qnode(dev)
        def circuit():
            return hqml.var(hqml.QuadX(0)), hqml.var(hqml.QuadP(0))

        hbar = 2
        dx, dp = circuit()
        assert dx * dp >= hbar / 2

    @pytest.mark.parametrize("alpha", (0.2, 0.5, 1.0, -1.0, -0.5, -0.2))
    def test_displacement_analytic(self, alpha):
        # Basic circuit that prepares |α> and checks the mean photon count
        # is <n> = |α|^2
        dev = qml.device(
            "hybrid.bosonicqiskit", wires=[0], qumodes=[0], max_fock_level=16
        )

        @qml.qnode(dev)
        def circuit(alpha):
            qml.Displacement(alpha, 0, 0)
            return hqml.expval(hqml.NumberOperator(0)), hqml.var(hqml.NumberOperator(0))

        expval, var = circuit(alpha)
        assert np.ndim(expval) == 0
        assert np.ndim(var) == 0
        assert np.isclose(expval, np.abs(alpha) ** 2)
        assert np.isclose(var, np.abs(alpha) ** 2)

    def test_displacement_on_multiqumode_system(self):
        alpha = 1
        lam = np.abs(alpha) ** 2
        truncation = FockTruncation.all_fock_space([0, 1], {0: 16, 1: 4})

        dev = qml.device("hybrid.bosonicqiskit", truncation=truncation)

        @qml.qnode(dev)
        def circuit(alpha):
            qml.Displacement(alpha, 0, 0)
            return hqml.expval(hqml.NumberOperator(0)), hqml.expval(
                hqml.NumberOperator(1)
            )

        n0, n1 = circuit(alpha)
        assert np.isclose(n0, lam)
        assert np.isclose(n1, 0)

    @pytest.mark.parametrize("alpha", (0.2, 0.5, 1.0, -1.0, -0.5, -0.2))
    def test_displacement_sampled(self, alpha):
        # Same test as above, but with finite samples. We'll test against the poisson distribution.
        # This tests finite sampling of unbounded cv operators (HasSpectrum)
        fock_levels = 16
        lam = np.abs(alpha) ** 2
        n_per_test = 5000
        repetitions = 10

        dev = qml.device(
            "hybrid.bosonicqiskit", wires=[0], qumodes=[0], max_fock_level=fock_levels
        )

        @partial(qml.set_shots, shots=repetitions * n_per_test)
        @qml.qnode(dev)
        def circuit(alpha):
            qml.Displacement(alpha, 0, 0)
            return hqml.expval(hqml.NumberOperator(0)), hqml.sample(
                hqml.NumberOperator(0)
            )

        # Rather than repeat the circuit `repetitions` times, which would be slower,
        # we just partition the shots ourselves into that many tests
        expval, samples = circuit(alpha)
        sample_set = samples.reshape(repetitions, n_per_test)

        # Sample format test
        assert sample_set.min() >= 0
        assert sample_set.max() <= fock_levels - 1

        rejections = 0
        for samples in sample_set:
            # Test overall distribution shape
            if (p_value := poisson_test(samples, lam)) < 0.05:
                rejections += 1

        # Check that we didn't reject more than a majority of our tests
        assert rejections / repetitions < 0.5

    @pytest.mark.parametrize("phi", (0, np.pi / 2, np.pi, 3 * np.pi / 2))
    def test_rotation_analytic(self, phi):
        alpha = 1.5
        truncation = FockTruncation.all_fock_space([0], {0: 16})

        dev = qml.device("hybrid.bosonicqiskit", truncation=truncation)

        @qml.qnode(dev)
        def circuit(alpha, phi):
            qml.Displacement(alpha, 0, 0)
            qml.Rotation(phi, 0)
            return hqml.expval(hqml.QuadX(0)), hqml.expval(hqml.QuadP(0))

        expval_x, expval_p = circuit(alpha, phi)
        expected_x = 2 * np.cos(phi) * alpha
        expected_p = 2 * np.sin(phi) * alpha
        assert np.isclose(expval_x, expected_x)
        assert np.isclose(expval_p, expected_p)

    @pytest.mark.parametrize("n", range(6))
    def test_create_fock_state_analytic(self, n):
        # Creates the state |0,n> through JC gates
        dev = qml.device(
            "hybrid.bosonicqiskit", wires=[0, "m0"], qumodes=["m0"], max_fock_level=8
        )

        @qml.qnode(dev)
        def circuit():
            for j in range(n):
                qml.X(0)
                hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, ["m0", 0])

            return hqml.expval(hqml.NumberOperator("m0")), hqml.expval(qml.Z(0))

        expval_n, expval_z = circuit()
        assert np.isclose(expval_n, n)
        assert np.isclose(expval_z, 1.0)

    def test_jc_analytic(self):
        dev = qml.device(
            "hybrid.bosonicqiskit",
            wires=list(range(4)),
            qumodes=[1, 3],
            max_fock_level=4,
        )

        @qml.qnode(dev)
        def circuit():
            # Put the first subsystem (qubit 0, qumode 2) in state |0>_Q |1>_B
            qml.X(0)
            hqml.JaynesCummings(np.pi / 2, np.pi / 2, [1, 0])

            # Put the second subsystem (qubit 1, qumode 3) in state |0>_Q |2>_B
            qml.X(2)
            hqml.JaynesCummings(np.pi / 2, np.pi / 2, [3, 2])
            qml.X(2)
            hqml.JaynesCummings(np.pi / (2 * np.sqrt(2)), np.pi / 2, [3, 2])

            # check qumodes in state |1>|2>
            return (
                hqml.expval(hqml.FockStateProjector([1, 2], [1, 3])),
                hqml.expval(hqml.NumberOperator(1)),
                hqml.expval(hqml.NumberOperator(3)),
            )

        expval, n1, n3 = circuit()
        assert np.isclose(n1, 1)
        assert np.isclose(n3, 2)
        assert np.isclose(expval, 1.0)

    def test_complex_fock_observable_analytic(self):
        # This is another coherent state, but this time we measure n + n^2, which
        # is diagonal in fock basis. However, this tests some of the static analysis
        alpha = 1.5
        lam = np.abs(alpha) ** 2
        truncation = FockTruncation.all_fock_space([0], {0: 16})

        dev = qml.device("hybrid.bosonicqiskit", truncation=truncation)

        @qml.qnode(dev)
        def circuit(alpha):
            qml.Displacement(alpha, 0, 0)
            return hqml.expval(hqml.NumberOperator(0) + hqml.NumberOperator(0) ** 2)

        n = circuit(alpha)
        expval_n = lam
        expval_n2 = lam + lam**2
        assert np.isclose(n, expval_n + expval_n2)

    def test_cv_swap(self):
        dev = qml.device(
            "hybrid.bosonicqiskit", wires=[0, 1], qumodes=[0, 1], max_fock_level=16
        )

        @qml.qnode(dev)
        def circuit(alpha):
            qml.Displacement(alpha, 0, 0)
            hqml.ModeSwap([0, 1])  # will get decomposed to beamsplitters
            qml.Displacement(-alpha, 0, 1)
            return hqml.expval(hqml.NumberOperator(0)), hqml.expval(
                hqml.NumberOperator(1)
            )

        alpha = 1.5
        n1, n2 = circuit(alpha)
        assert np.isclose(n1, 0)
        assert np.isclose(n2, 0)

    # Fixme: this test fails because constructing ExpectationMP infers a schema, but the schemas
    # for n and x are different. However, technically in an analytic simulation of bosonic qiskit,
    # it could handle this just fine. Maybe we need to be more deliberate about where verification
    # and static analysis happen?
    @pytest.mark.skip
    def test_complex_multibasis_observable_analytic(self):
        # This is another coherent state, but this time we measure n + x
        alpha = 1.5
        lam = np.abs(alpha) ** 2
        truncation = FockTruncation.all_fock_space([0], {0: 16})

        dev = qml.device("hybrid.bosonicqiskit", truncation=truncation)

        @qml.qnode(dev)
        def circuit(alpha):
            qml.Displacement(alpha, 0, 0)
            return hqml.expval(hqml.NumberOperator(0) + hqml.QuadX(0))

        n = circuit(alpha)
        expval_n = lam
        expval_x = 2 * alpha
        assert np.isclose(n, expval_n + expval_x)

    # Fixme: this test fails, it is supposed to have an SQR gate but that's not
    # yet implemented in bosonic qiskit
    @pytest.mark.skip
    @pytest.mark.parametrize("alpha", (1.0, -1.0, 2.0, -2.0))
    def test_cat_state_readout(self, alpha):
        fock_levels = 16
        n_per_test = 5000
        repetitions = 10

        dev = qml.device(
            "hybrid.bosonicqiskit", wires=[0], qumodes=[0], max_fock_level=fock_levels
        )

        @partial(qml.set_shots, shots=repetitions * n_per_test)
        @qml.qnode(dev)
        def circuit(alpha):
            # Put the qumode into state |α> + |-α>
            qml.H(0)
            hqml.ConditionalDisplacement(alpha, 0, wires=[1, 0])
            qml.H(0)

            # Now use ancilliary qubit to read it out with a phase kickback
            qml.Displacement(alpha, 0, 1)  # |0> + |2α>
            qml.H(2)
            hqml.SelectiveNumberArbitraryPhase(np.pi / 2, 0, wires=[1, 2])
            qml.H(2)

            return hqml.sample(qml.PauliZ(2))

        eigvals = circuit(alpha)
        sample_set = (1 - eigvals) // 2
        sample_set = sample_set.reshape(repetitions, n_per_test)
        a2 = np.abs(alpha) ** 2
        p = 0.5 * (1 + np.exp(-4 * a2)) + np.exp(-2 * a2)

        rejections = 0
        for samples in sample_set:
            successes = int(samples.sum())
            if (
                pvalue := scipy.stats.binomtest(successes, n_per_test, p).pvalue
            ) < 0.05:
                rejections += 1

        assert rejections / repetitions < 0.5
