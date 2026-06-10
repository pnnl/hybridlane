import jax
import numpy as np
import pennylane as qp
import pytest

import hybridlane as hl
from hybridlane.devices.default_hybrid.state_prep import coherent_state
from hybridlane.measurements import BasisSchema, SampleResult
from hybridlane.sa import ComputationalBasis
from test.util import poisson_test


@pytest.mark.all_interfaces
@pytest.mark.integration
class TestStateMeasurements:
    def test_density_matrix(self, like):
        fock_levels = 4
        dev = qp.device("default.hybrid", fock_level=fock_levels)

        # Prepares a cat state with residual entanglement with a qubit
        @qp.qnode(dev, interface=like)
        def circuit():
            qp.H(0)
            hl.CD(0.123, 0, wires=[0, 1])
            return hl.density_matrix(wires=1)

        rho = circuit()
        assert hl.math.get_interface(rho) == like
        assert rho.shape == (fock_levels, fock_levels)

        # Check density matrix properties
        assert hl.math.linalg.trace(rho) == pytest.approx(1)
        assert hl.math.linalg.trace(hl.math.linalg.matrix_power(rho, 2)) < 1


@pytest.mark.all_interfaces
@pytest.mark.integration
class TestAnalyticCircuits:
    def test_vacuum_expval(self, like):
        dev = qp.device("default.hybrid", fock_level=16)

        @qp.qnode(dev, interface=like)
        def circuit():
            return hl.expval(hl.N(0)), hl.expval(hl.X(0))

        result = circuit()
        assert hl.math.get_deep_interface(result) == like
        assert result == pytest.approx([0, 0])

    def test_vacuum_var(self, like):
        dev = qp.device("default.hybrid", fock_level=16)

        @qp.qnode(dev, interface=like)
        def circuit():
            return hl.var(hl.N(0))

        result = circuit()
        assert hl.math.get_interface(result) == like
        assert result == pytest.approx(0)

    def test_heisenberg_uncertainty(self, like):
        dev = qp.device("default.hybrid", fock_level=16)

        @qp.qnode(dev, interface=like)
        def circuit():
            return hl.var(hl.X(0)), hl.var(hl.P(0))

        dx, dp = circuit()
        assert hl.math.get_interface(dx) == like
        assert hl.math.get_interface(dp) == like
        assert hl.math.sqrt(dx * dp) - 1 / 2 == pytest.approx(0)

    def test_bell_state_stabilizer(self, like):
        dev = qp.device("default.hybrid", fock_level=16)

        @qp.qnode(dev, interface=like)
        def circuit():
            qp.H(0)
            qp.CNOT(wires=[0, 1])
            return hl.expval(qp.X(0) @ qp.X(1)), hl.expval(qp.Z(0) @ qp.Z(1))

        expvals = circuit()
        assert hl.math.get_deep_interface(expvals) == like
        assert expvals == pytest.approx([1, 1])

    def test_coherent_state(self, like):
        fock_level = 8
        dev = qp.device("default.hybrid", fock_level=fock_level)

        @qp.qnode(dev, interface=like)
        def circuit(a):
            hl.D(a, 0, wires=0)
            return hl.state()

        a = hl.math.array(0.123, like=like)
        state = circuit(a)
        assert hl.math.get_interface(state) == like
        assert state.shape == (fock_level,)
        assert state == pytest.approx(coherent_state(0.123, 8), abs=1e-9)

    def test_cat_state_readout(self, like):
        dev = qp.device("default.hybrid", fock_level=32)

        @qp.qnode(dev, interface=like)
        def circuit(alpha):
            # Put the qumode into state |α> + |-α>, which acts like |0L> + |1L>
            qp.CatState(alpha, 0, 0, wires=0)

            # Now use ancilliary qubit to read it out with a phase kickback
            hl.D(alpha, 0, 0)  # |0> + |2α>
            qp.H(1)
            hl.SQR(np.pi, np.pi / 2, 0, wires=[1, 0])  # Ry(pi)|0><0|
            qp.H(1)

            return hl.expval(qp.Z(1))

        # With a fock truncation of 32, we expect the coherent state distribution to follow
        # a poisson distribution. poisson.sf(32, 3^2) ~ 1e-10, so this truncation should
        # suffice.
        alpha = hl.math.array(3.0, like=like)
        expval = circuit(alpha)
        assert hl.math.get_interface(expval) == like
        assert expval == pytest.approx(0, abs=1e-7)  # 50-50

    def test_cv_swap(self, like):
        dev = qp.device("default.hybrid", fock_level=16)

        @qp.qnode(dev, interface=like)
        def circuit(alpha):
            hl.D(alpha, 0, 0)
            hl.ModeSwap([0, 1])
            hl.D(-alpha, 0, 1)
            return hl.expval(hl.N(0)), hl.expval(hl.N(1))

        alpha = 1.5
        n1, n2 = circuit(alpha)
        assert hl.math.get_interface(n1) == like
        assert hl.math.get_interface(n2) == like
        assert n1 == pytest.approx(0)
        assert n2 == pytest.approx(0)


@pytest.mark.integration
@pytest.mark.all_interfaces
class TestFiniteCircuits:
    def test_sample_coherent_state(self, like):
        fock_levels = 32
        n_per_test = 5000
        repetitions = 10

        dev = qp.device("default.hybrid", fock_level=fock_levels)

        @qp.set_shots(repetitions * n_per_test)
        @qp.qnode(dev, interface=like)
        def circuit(alpha):
            hl.D(alpha, 0, 0)
            return hl.sample(hl.N(0))

        if like == "jax":
            pytest.xfail(reason="JAX jit doesn't work with sample yet")
            circuit = jax.jit(circuit)

        # Rather than repeat the circuit `repetitions` times, which would be slower,
        # we just partition the shots ourselves into that many tests
        alpha = hl.math.array(3.0, like=like)
        lam = alpha**2
        samples = circuit(alpha)

        assert hl.math.get_interface(samples) == like
        assert hl.math.get_dtype_name(samples) == "int64"
        assert samples.shape == (repetitions * n_per_test,)

        sample_set = hl.math.reshape(samples, (repetitions, n_per_test))

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

    def test_coherent_state_expval(self, like):
        fock_levels = 32

        dev = qp.device("default.hybrid", fock_level=fock_levels, seed=42)

        @qp.set_shots(100_000)
        @qp.qnode(dev, interface=like)
        def circuit(alpha):
            hl.D(alpha, 0, 0)
            return hl.expval(hl.N(0))

        if like == "jax":
            circuit = jax.jit(circuit)

        alpha = hl.math.array(3.0, like=like)
        lam = alpha**2
        expval = circuit(alpha)

        assert hl.math.get_interface(expval) == like
        assert hl.math.get_dtype_name(expval) == "float64"
        assert expval.shape == ()
        assert expval == pytest.approx(lam, abs=0.01)

    def test_cat_state_expval(self, like):
        fock_levels = 32

        dev = qp.device("default.hybrid", fock_level=fock_levels, seed=42)

        @qp.set_shots(100_000)
        @qp.qnode(dev, interface=like)
        def circuit(alpha):
            qp.H(0)
            hl.CD(alpha, 0, wires=(0, 1))
            qp.H(0)
            return hl.expval(qp.Z(0) @ hl.N(1)), hl.expval(qp.Z(0)), hl.expval(hl.N(1))

        if like == "jax":
            circuit = jax.jit(circuit)

        alpha = hl.math.array(3.0, like=like)
        lam = alpha**2
        zn, z, n = circuit(alpha)

        for result in (zn, z, n):
            assert hl.math.get_interface(result) == like
            assert hl.math.get_dtype_name(result) == "float64"
            assert result.shape == ()

        assert zn == pytest.approx(0, abs=0.1)  # equal contribution cancels out
        assert z == pytest.approx(0, abs=0.1)  # equal contribution cancels out
        assert n == pytest.approx(lam, abs=0.1)

    def test_qpe(self, like):
        U = hl.R(0.5, wires="m")

        dev = qp.device("default.hybrid", fock_level=8)

        @qp.transforms.decompose(
            gate_set={
                hl.Red,
                hl.Blue,
                hl.ConditionalRotation,
                hl.Rotation,
                qp.RZ,
                qp.CRZ,
                qp.CNOT,
                qp.H,
                qp.ControlledPhaseShift,
            },
        )
        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit(n_bits: int):
            hl.FockState(4, wires=("q", "m"))
            estimation_wires = range(n_bits)
            qp.QuantumPhaseEstimation(U, estimation_wires=estimation_wires)

            schema = BasisSchema({estimation_wires: ComputationalBasis.Discrete})
            return hl.sample(schema=schema)

        if like == "jax":
            pytest.xfail(reason="JAX jit doesn't work with sample yet")
            circuit = jax.jit(circuit, static_argnums=0)

        n_bits = 3
        samples = circuit(n_bits)
        assert isinstance(samples, SampleResult)
        assert len(samples.data) == n_bits
        assert samples.shape == (10,)

        for bits in samples.data.values():
            assert hl.math.get_deep_interface(bits) == like
            assert hl.math.get_dtype_name(bits) == "int64"

    def test_fock_state_var(self, like):
        fock_levels = 8

        dev = qp.device("default.hybrid", fock_level=fock_levels)

        @qp.set_shots(100_000)
        @qp.qnode(dev, interface=like)
        def circuit(n):
            qp.FockState(n, wires=0)
            return hl.var(hl.N(0))

        if like == "jax":
            circuit = jax.jit(circuit, static_argnums=0)

        for n in range(fock_levels):
            var = circuit(n)
            assert hl.math.get_interface(var) == like
            assert hl.math.get_dtype_name(var) == "float64"
            assert var.shape == ()
            assert var == pytest.approx(0)

    def test_bell_state_stabilizer(self, like):
        dev = qp.device("default.hybrid", fock_level=16)

        @qp.set_shots(10)
        @qp.qnode(dev, interface=like)
        def circuit():
            qp.H(0)
            qp.CNOT(wires=[0, 1])
            return hl.expval(qp.X(0) @ qp.X(1)), hl.expval(qp.Z(0) @ qp.Z(1))

        if like == "jax":
            circuit = jax.jit(circuit)

        expvals = circuit()
        assert hl.math.get_deep_interface(expvals) == like
        assert expvals == pytest.approx([1, 1])

    def test_cat_state_expval_hamiltonian(self, like):
        fock_levels = 32

        dev = qp.device("default.hybrid", fock_level=fock_levels, seed=42)

        @qp.set_shots(100_000)
        @qp.qnode(dev, interface=like)
        def circuit(alpha):
            qp.H(0)
            hl.CD(alpha, 0, wires=(0, 1))
            qp.H(0)
            return hl.expval(qp.Z(0) + 0.1 * hl.N(1))

        if like == "jax":
            circuit = jax.jit(circuit)

        alpha = hl.math.array(3.0, like=like)
        lam = alpha**2
        expval = circuit(alpha)

        assert hl.math.get_interface(expval) == like
        assert hl.math.get_dtype_name(expval) == "float64"
        assert expval.shape == ()
        expval == pytest.approx(0.1 * lam, abs=0.1)
