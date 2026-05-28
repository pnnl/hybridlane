# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import sys
from functools import partial

import numpy as np
import pennylane as qp
import pytest
from pennylane.exceptions import DeviceError

import hybridlane as hl
from hybridlane.measurements import FockTruncation
from hybridlane.sa.exceptions import StaticAnalysisError

from ...util import poisson_test


@pytest.mark.unit
def test_package_works_without_bosonic_qiskit(monkeypatch):
    monkeypatch.delitem(sys.modules, "bosonic_qiskit", raising=False)
    import hybridlane  # noqa: F401


@pytest.mark.bq
class TestBosonicQiskitDevice:
    @pytest.mark.unit
    def test_device_is_registered(self):
        from hybridlane.devices import BosonicQiskitDevice

        dev = qp.device("bosonicqiskit.hybrid")
        assert isinstance(dev, BosonicQiskitDevice)

    @pytest.mark.unit
    def test_non_power_of_two_truncation(self):
        trunc = FockTruncation.all_fock_space([0, 1], {0: 2, 1: 7})
        dev = qp.device("bosonicqiskit.hybrid", truncation=trunc)

        @qp.qnode(dev)
        def circuit():
            hl.ConditionalDisplacement(1.0, 0, [0, 1])
            return hl.expval(hl.NumberOperator(1))

        with pytest.raises(DeviceError):
            circuit()

    @pytest.mark.unit
    def test_no_inferrable_truncation(self):
        # This circuit has a qumode that should be detected through static analysis,
        # but no truncation is provided.
        dev = qp.device("bosonicqiskit.hybrid")

        @qp.qnode(dev)
        def circuit():
            qp.Rotation(0.5, 0)
            return hl.expval(hl.NumberOperator(0))

        with pytest.raises(DeviceError):
            circuit()

    @pytest.mark.unit
    def test_infer_qubits(self):
        # This circuit should be detected as all qubit and therefore automagically
        # derive a truncation of 2 for each qubit. It'll then fail because
        # we only simulate hybrid programs.
        dev = qp.device("bosonicqiskit.hybrid")

        @qp.qnode(dev)
        def circuit():
            qp.H(0)
            return hl.expval(qp.Z(0) @ qp.X(1))

        with pytest.raises(DeviceError):
            circuit()

    @pytest.mark.unit
    def test_wires_aliased_by_operation(self):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=8)

        @qp.qnode(dev)
        def circuit():
            hl.ConditionalDisplacement(
                1.0, 0, [0, 1]
            )  # wire 0 established as a qubit here
            return hl.expval(hl.NumberOperator(0))  # measure wire 0 in fock basis

        with pytest.raises(StaticAnalysisError):
            circuit()

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "obs",
        (
            hl.NumberOperator(0) @ qp.PauliZ(0),
            hl.NumberOperator(0) + qp.PauliZ(0),
            qp.s_prod(0.5, hl.NumberOperator(0) @ qp.PauliZ(0)),
            qp.s_prod(0.5, hl.NumberOperator(0) + qp.PauliZ(0)),
        ),
    )
    def test_wires_aliased_by_observable(self, obs):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=8)

        @qp.qnode(dev)
        def circuit():
            return hl.expval(obs)

        with pytest.raises(StaticAnalysisError):
            circuit()

    @pytest.mark.integration
    def test_units(self):
        alpha = 1.5
        truncation = FockTruncation.all_fock_space([0], {0: 16})

        def circuit(alpha):
            qp.Displacement(alpha, 0, 0)
            return hl.expval(hl.QuadX(0)), hl.expval(hl.QuadP(0))

        units = "standard"
        dev = qp.device("bosonicqiskit.hybrid", truncation=truncation, units=units)
        qnode = qp.QNode(circuit, dev)
        expval_x, expval_p = qnode(alpha)
        expected_x = np.sqrt(2) * alpha
        expected_p = 0
        assert np.isclose(expval_x, expected_x)
        assert np.isclose(expval_p, expected_p)

        units = "wigner"
        dev = qp.device("bosonicqiskit.hybrid", truncation=truncation, units=units)
        qnode = qp.QNode(circuit, dev)
        expval_x, expval_p = qnode(alpha)
        expected_x = alpha
        expected_p = 0
        assert np.isclose(expval_x, expected_x)
        assert np.isclose(expval_p, expected_p)


@pytest.mark.bq
class TestOperations:
    @pytest.mark.integration
    def test_fockstatevector(self):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=4)

        # Put qumode 0 in state |2> and put qumode 1 in a superposition of |0> and |1>
        state0 = np.zeros((4,), dtype=complex)
        state0[2] = 1.0
        state1 = np.zeros((4,), dtype=complex)
        state1[0] = 1 / np.sqrt(2)
        state1[1] = 1 / np.sqrt(2)
        state = np.kron(state0, state1).reshape(4, 4)

        @qp.qnode(dev)
        def circuit():
            qp.FockStateVector(state, wires=[0, 1])
            return (hl.expval(hl.N(0)), hl.expval(hl.N(1)))

        n0, n1 = circuit()
        assert np.isclose(n0, 2)
        assert np.isclose(n1, 0.5)

    @pytest.mark.integration
    @pytest.mark.parametrize("alpha", (0.2, 0.5, 1.0, -1.0, -0.5, -0.2))
    def test_displacement(self, alpha):
        # Basic circuit that prepares |α> and checks the mean photon count
        # is <n> = |α|^2
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit(alpha):
            qp.Displacement(alpha, 0, 0)
            return hl.expval(hl.NumberOperator(0)), hl.var(hl.NumberOperator(0))

        expval, var = circuit(alpha)
        assert np.ndim(expval) == 0
        assert np.ndim(var) == 0
        assert np.isclose(expval, np.abs(alpha) ** 2)
        assert np.isclose(var, np.abs(alpha) ** 2)

    @pytest.mark.integration
    def test_multiqumode_displacement(self):
        alpha = 1
        lam = np.abs(alpha) ** 2
        truncation = FockTruncation.all_fock_space([0, 1], {0: 16, 1: 4})

        dev = qp.device("bosonicqiskit.hybrid", truncation=truncation)

        @qp.qnode(dev)
        def circuit(alpha):
            qp.Displacement(alpha, 0, 0)
            return hl.expval(hl.NumberOperator(0)), hl.expval(hl.NumberOperator(1))

        n0, n1 = circuit(alpha)
        assert np.isclose(n0, lam)
        assert np.isclose(n1, 0)

    @pytest.mark.integration
    @pytest.mark.parametrize("phi", (0, np.pi / 2, np.pi, 3 * np.pi / 2))
    def test_rotation(self, phi):
        alpha = 1.5
        truncation = FockTruncation.all_fock_space([0], {0: 16})

        dev = qp.device("bosonicqiskit.hybrid", truncation=truncation)

        @qp.qnode(dev)
        def circuit(alpha, phi):
            qp.Displacement(alpha, 0, 0)
            qp.Rotation(phi, 0)
            return hl.expval(hl.QuadX(0)), hl.expval(hl.QuadP(0))

        expval_x, expval_p = circuit(alpha, phi)
        expected_x = np.sqrt(2) * np.cos(phi) * alpha
        expected_p = np.sqrt(2) * np.sin(phi) * alpha
        assert np.isclose(expval_x, expected_x)
        assert np.isclose(expval_p, expected_p)

    @pytest.mark.integration
    def test_coherent_state(self):
        alpha = 1.5
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit(alpha):
            qp.CoherentState(alpha, 0, wires=0)
            return hl.expval(hl.N(0))

        expval = circuit(alpha)
        assert np.isclose(expval, np.abs(alpha) ** 2)

    @pytest.mark.integration
    def test_cat_state(self):
        alpha = 1.5
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit(alpha):
            qp.CatState(alpha, 0, p=1, wires=0)
            qp.change_op_basis(qp.H(1), hl.CP([1, 0]))
            return hl.expval(qp.Z(1))

        expval = circuit(alpha)
        assert np.isclose(expval, -1)

    @pytest.mark.integration
    def test_bell_state_prep(self):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        state = np.zeros((4,), dtype=complex)
        state[1] = 1 / np.sqrt(2)
        state[2] = 1 / np.sqrt(2)

        @qp.qnode(dev)
        def circuit():
            # Prepare a bell state and then measure a vacuum qumode just because
            # bosonic qiskit requires 1 qumode
            qp.StatePrep(state, wires=[0, 1])
            return hl.expval(qp.X(0) @ qp.X(1)), hl.expval(hl.N(2))

        stabilizer, _ = circuit()
        assert np.isclose(stabilizer, 1)

    @pytest.mark.integration
    def test_qubit_basis_state(self):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit():
            qp.BasisState([1, 0, 0], wires=[0, 1, 2])
            hl.D(1, 1, wires=3)  # again dummy qumode
            return hl.expval(qp.Z(0))

        expval = circuit()
        assert np.isclose(expval, -1)


@pytest.mark.bq
class TestObservableMeasurements:
    @pytest.mark.integration
    def test_vacuum_expval(self):
        # The simplest test you could do, checking the vacuum state |0> has <n> = 0
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit():
            return hl.expval(hl.NumberOperator(0))

        result = circuit()
        assert np.ndim(result) == 0
        assert np.isclose(result, 0)

    @pytest.mark.integration
    def test_vacuum_var(self):
        # Checking the vacuum state |0> has Var(n) = 0 since it's a definite eigenstate
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit():
            return hl.var(hl.NumberOperator(0))

        result = circuit()
        assert np.ndim(result) == 0
        assert np.isclose(result, 0)

    @pytest.mark.integration
    def test_heisenberg_uncertainty(self):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit():
            return hl.var(hl.QuadX(0)), hl.var(hl.QuadP(0))

        dx, dp = circuit()
        assert np.sqrt(dx * dp) >= 1 / 2

    @pytest.mark.integration
    @pytest.mark.parametrize("alpha", (0.2, 0.5, 1.0, -1.0, -0.5, -0.2))
    def test_sample_coherent_state(self, alpha):
        fock_levels = 16
        lam = np.abs(alpha) ** 2
        n_per_test = 5000
        repetitions = 10

        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=fock_levels)

        @partial(qp.set_shots, shots=repetitions * n_per_test)
        @qp.qnode(dev)
        def circuit(alpha):
            qp.Displacement(alpha, 0, 0)
            return hl.sample(hl.NumberOperator(0))

        # Rather than repeat the circuit `repetitions` times, which would be slower,
        # we just partition the shots ourselves into that many tests
        samples = circuit(alpha)
        sample_set = samples.reshape(repetitions, n_per_test)

        # Sample format test
        assert sample_set.min() >= 0
        assert sample_set.max() <= fock_levels - 1

        rejections = 0
        for samples in sample_set:
            # Test overall distribution shape
            if poisson_test(samples, lam) < 0.05:
                rejections += 1

        # Check that we didn't reject more than a majority of our tests
        assert rejections / repetitions < 0.5

    @pytest.mark.integration
    def test_complex_fock_observable_analytic(self):
        # This is another coherent state, but this time we measure n + n^2, which
        # is diagonal in fock basis. However, this tests some of the static analysis
        alpha = 1.5
        lam = np.abs(alpha) ** 2
        truncation = FockTruncation.all_fock_space([0], {0: 16})

        dev = qp.device("bosonicqiskit.hybrid", truncation=truncation)

        @qp.qnode(dev)
        def circuit(alpha):
            qp.Displacement(alpha, 0, 0)
            return hl.expval(hl.NumberOperator(0) + hl.NumberOperator(0) ** 2)

        n = circuit(alpha)
        expval_n = lam
        expval_n2 = lam + lam**2
        assert np.isclose(n, expval_n + expval_n2)

    # Fixme: this test fails because constructing ExpectationMP infers a schema, but
    # the schemas for n and x are different. However, technically in an analytic
    # simulation of bosonic qiskit, it could handle this just fine. Maybe we need to be
    # more deliberate about where verification and static analysis happen?
    @pytest.mark.integration
    @pytest.mark.xfail
    def test_complex_multibasis_observable_analytic(self):
        # This is another coherent state, but this time we measure n + x
        alpha = 1.5
        lam = np.abs(alpha) ** 2
        truncation = FockTruncation.all_fock_space([0], {0: 16})

        dev = qp.device("bosonicqiskit.hybrid", truncation=truncation)

        @qp.qnode(dev)
        def circuit(alpha):
            qp.Displacement(alpha, 0, 0)
            return hl.expval(hl.NumberOperator(0) + hl.QuadX(0))

        n = circuit(alpha)
        expval_n = lam
        expval_x = 2 * alpha
        assert np.isclose(n, expval_n + expval_x)


@pytest.mark.bq
@pytest.mark.integration
class TestStateMeasurements:
    @pytest.mark.parametrize(["wires", "state_index"], [([0, 1], 1), ([1, 0], 2)])
    def test_statevector_with_wire_flips(self, wires, state_index):
        fock_levels = 4
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=fock_levels, wires=2)

        @qp.qnode(dev)
        def circuit():
            hl.FockState(
                1, wires
            )  # set mode to 1 using wire[0] as qubit control and wire[1] as qumode
            return (
                hl.state(),
                hl.expval(hl.NumberOperator(wires[1])),
            )

        state, num = circuit()
        assert np.isclose(num, 1)
        target = np.zeros((8,), dtype=complex)
        target[state_index] = 1.0
        assert np.allclose(state, target)

    @pytest.mark.parametrize(
        ["wires", "state_index"],
        [
            ([0, 1, 2], 6),
            ([0, 2, 1], 9),
            ([1, 0, 2], 10),
            ([1, 2, 0], 17),
            ([2, 0, 1], 12),
            ([2, 1, 0], 18),
        ],
    )
    def test_statevector_with_more_wires(self, wires, state_index):
        fock_levels = 4
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=fock_levels, wires=3)

        @qp.qnode(dev)
        def circuit():
            # always assume wire[0] is qubit control and wire[1] and wire[2] is qumode
            hl.FockState(  # set mode to 1 using wire[0] as qubit control and wire[1] as qumode
                1, [wires[0], wires[1]]
            )
            hl.FockState(  # set mode to 1 using wire[0] as qubit control and wire[2] as qumode
                2, [wires[0], wires[2]]
            )
            return (
                hl.state(),
                hl.expval(hl.NumberOperator(wires[1])),
                hl.expval(hl.NumberOperator(wires[2])),
            )

        state, num1, num2 = circuit()
        assert np.isclose(num1, 1)
        assert np.isclose(num2, 2)
        target = np.zeros((32,), dtype=complex)
        target[state_index] = 1.0
        assert np.allclose(state, target)

    def test_density_matrix(self):
        fock_levels = 4
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=fock_levels)

        # Prepares a cat state with residual entanglement with a qubit
        @qp.qnode(dev)
        def circuit():
            qp.H(0)
            hl.CD(0.123, 0, wires=[0, 1])
            return hl.density_matrix(wires=1)

        rho = circuit()
        assert rho.shape == (fock_levels, fock_levels)

        # Check density matrix properties
        assert hl.math.linalg.trace(rho) == pytest.approx(1)
        assert hl.math.linalg.trace(hl.math.linalg.matrix_power(rho, 2)) < 1


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.bq
class TestIntegration:
    @pytest.mark.parametrize("n", range(6))
    def test_create_fock_state_analytic(self, n):
        # Creates the state |0,n> through JC gates
        dev = qp.device("bosonicqiskit.hybrid", wires=[0, "m0"], max_fock_level=8)

        @qp.qnode(dev)
        def circuit():
            for j in range(n):
                qp.X(0)
                hl.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, "m0"])

            return hl.expval(hl.NumberOperator("m0")), hl.expval(qp.Z(0))

        expval_n, expval_z = circuit()
        assert np.isclose(expval_n, n)
        assert np.isclose(expval_z, 1.0)

    def test_jc_analytic(self):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=4)

        @qp.qnode(dev)
        def circuit():
            # Put the first subsystem (qubit 0, qumode 1) in state |0>_Q |1>_B
            qp.X(0)
            hl.JaynesCummings(np.pi / 2, np.pi / 2, [0, 1])

            # Put the second subsystem (qubit 2, qumode 3) in state |0>_Q |2>_B
            qp.X(2)
            hl.JaynesCummings(np.pi / 2, np.pi / 2, [2, 3])
            qp.X(2)
            hl.JaynesCummings(np.pi / (2 * np.sqrt(2)), np.pi / 2, [2, 3])

            # check qumodes in state |1>|2>
            return (
                qp.expval(
                    hl.FockStateProjector([1, 2], [1, 3])
                ),  # check that from_pennylane transform handles it
                hl.expval(hl.NumberOperator(1)),
                hl.expval(hl.NumberOperator(3)),
            )

        expval, n1, n3 = circuit()
        assert np.isclose(n1, 1)
        assert np.isclose(n3, 2)
        assert np.isclose(expval, 1.0)

    def test_cv_swap(self):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit(alpha):
            qp.Displacement(alpha, 0, 0)
            hl.ModeSwap([0, 1])  # will get decomposed to beamsplitters
            qp.Displacement(-alpha, 0, 1)
            return hl.expval(hl.NumberOperator(0)), hl.expval(hl.NumberOperator(1))

        alpha = 1.5
        n1, n2 = circuit(alpha)
        assert np.isclose(n1, 0)
        assert np.isclose(n2, 0)

    @pytest.mark.parametrize("alpha", (2.0, -2.0))
    def test_cat_state_readout(self, alpha):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit(alpha):
            # Put the qumode into state |α> + |-α>, which acts like |0L> + |1L>
            qp.H(0)
            hl.CD(alpha, 0, wires=[0, 1])
            qp.H(0)

            # Now use ancilliary qubit to read it out with a phase kickback
            qp.Displacement(alpha, 0, 1)  # |0> + |2α>
            qp.H(2)
            hl.SQR(np.pi, np.pi / 2, 0, wires=[2, 1])  # Ry(pi)|0><0|
            qp.H(2)

            return hl.expval(qp.Z(2))

        z = circuit(alpha)
        assert np.isclose(z, 0, atol=1e-7)
