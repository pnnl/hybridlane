import numpy as np
import pennylane as qp
import pytest

import hybridlane as hl
from hybridlane.devices.default_hybrid.state_prep import coherent_state


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
