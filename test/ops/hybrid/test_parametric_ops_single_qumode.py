# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qp
import pytest
from pennylane.tape.qscript import QuantumScript

import hybridlane as hl


@pytest.mark.unit
class TestConditionalRotation:
    def test_init(self):
        op = hl.ConditionalRotation(0.5, wires=[0, 1])
        assert op.name == "ConditionalRotation"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.ConditionalRotation(0.5, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.ConditionalRotation)
        assert adj_op.parameters[0] == -0.5

    def test_pow(self):
        op = hl.ConditionalRotation(0.5, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.ConditionalRotation)
        assert pow_op[0].parameters[0] == 1.0

    def test_simplify(self):
        op = hl.ConditionalRotation(0, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

        op = hl.ConditionalRotation(1e-9, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.ConditionalRotation(0.5, wires=[0, 1])
        assert op.label() == "CR"

    def test_fock_matrix(self):
        theta = math.pi / 2
        op = hl.ConditionalRotation(theta, wires=[0, 1])
        dim = 3
        matrix = op.fock_matrix({0: 2, 1: dim})
        r = hl.Rotation.compute_fock_matrix((dim,), theta / 2)
        rd = hl.math.dag(r)
        expected = hl.math.block_diag([r, rd])
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(math.pi / 2)
        op = hl.ConditionalRotation(theta, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hl.math.eye(8, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(theta):
            op = hl.CR(theta, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        theta = jnp.array(math.pi / 4)
        f(theta)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dim = 4

        # Should output |0>sin(xn/2) + |1>sin(-xn/2)
        def f(theta):
            op = hl.CR(theta, wires=(0, 1))
            mat = op.fock_matrix({0: 2, 1: dim})
            return (mat @ jnp.ones(2 * dim)).imag.reshape(2, dim)  # ty:ignore[unsupported-operator]

        theta = jnp.array(math.pi / 4)
        n = jnp.arange(dim)
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(theta)
        expected_grad = -n / 2 * jnp.cos(theta / 2 * n)
        assert grad[0] == pytest.approx(expected_grad)
        assert grad[1] == pytest.approx(-expected_grad)


@pytest.mark.unit
class TestSelectiveQubitRotation:
    def test_init(self):
        op = hl.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        assert op.name == "SelectiveQubitRotation"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.SelectiveQubitRotation)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.SelectiveQubitRotation)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.SelectiveQubitRotation(0, 0.3, 1, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.SQR(0.5, 0.3, 1, wires=[0, 1])
        assert op.label() == "SQR_{1}"

    def test_fock_matrix_unitary(self):
        op = hl.SelectiveQubitRotation(0.5, 0.3, 0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hl.math.eye(8)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hl.SelectiveQubitRotation(theta, phi, 0, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hl.math.eye(8, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.SQR(*x, 1, wires=(0, 1))  # ty:ignore[parameter-already-assigned]
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, 0])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 2, 1: 4}
        # |1,1>
        state = hl.math.kron(jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0, 0.0, 0.0]))
        z = hl.math.expand_matrix(qp.Z(0).matrix(), (0,), wire_order=(0, 1), wire_dims=dims)
        z = hl.math.asarray(z, like=state)

        def f(x, n):
            op = hl.SQR(*x, n, wires=(0, 1))  # ty:ignore[parameter-already-assigned]
            mat = op.fock_matrix(dims)
            return hl.math.expectation_value(z, mat @ state).real

        x = jnp.array([0.123, 0])
        grad_fn = hl.math.grad(f)

        # Because the qumode is state |1>, setting the SQR gate to act on |0> does nothing
        grad = grad_fn(x, 0)
        assert grad == pytest.approx(0)

        # Now should act non-trivially. Because qubit is state |1>, continuing to rotate will
        # boost the value <Z>
        grad = grad_fn(x, 1)
        assert grad[0] > 0


@pytest.mark.unit
class TestJaynesCummings:
    def test_init(self):
        op = hl.JaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "JaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.JaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.JaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.JaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.JaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.JaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

        # JC is not periodic w.r.t. theta
        op = hl.JC(5 * math.pi, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert simplified_op.parameters == pytest.approx(op.parameters)

    def test_label(self):
        op = hl.JaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.label() == "JC"

    def test_fock_matrix_unitary(self):
        op = hl.JaynesCummings(0.5, 0.3, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hl.math.eye(8)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    def test_fock_matrix_action(self):
        # Test that we can move one quanta from qubit -> qumode
        #   JC(pi/2, 0)|1, 0> = |0, 1>
        dim = 4
        op = hl.JaynesCummings(math.pi / 2, 0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: dim})

        qumode_state = hl.math.zeros(dim)
        qumode_state[0] = 1
        qubit_state = hl.math.asarray([0, 1])
        state = hl.math.kron(qubit_state, qumode_state)
        evolved_state = matrix @ state

        expected_qumode_state = hl.math.zeros_like(qumode_state)
        expected_qumode_state[1] = 1
        expected_qubit_state = hl.math.asarray([1, 0])
        expected_state = hl.math.kron(expected_qubit_state, expected_qumode_state)

        assert hl.math.fidelity_statevector(evolved_state, expected_state) == pytest.approx(1)

        # Evolve again to obtain the original state
        evolved_state = matrix @ evolved_state
        assert hl.math.fidelity_statevector(evolved_state, state) == pytest.approx(1)

    def test_fock_matrix_commutes(self):
        # Check that the matrix conserves total excitation
        #   [JC, n + |1><1|] = 0
        dim = 5
        wire_dims = {0: 2, 1: dim}
        p1 = hl.math.asarray([[0, 0], [0, 1]])
        p1 = hl.math.expand_matrix(p1, (0,), wire_dims=wire_dims, wire_order=(0, 1))
        n = hl.N(1).fock_matrix(wire_dims, wire_order=(0, 1))
        obs = p1 + n

        op = hl.JaynesCummings(0.5, 0.3, wires=[0, 1])
        matrix = op.fock_matrix(wire_dims)
        comm = obs @ matrix - matrix @ obs
        assert comm == pytest.approx(0)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hl.JaynesCummings(theta, phi, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hl.math.eye(8, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.JC(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, 0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 2, 1: 4}
        # |1,0>
        state = hl.math.kron(jnp.array([0.0, 1.0]), jnp.array([1.0, 0.0, 0.0, 0.0]))
        n = hl.N(1).fock_matrix(wire_dims=dims, wire_order=(0, 1))

        def f(x):
            op = hl.JC(*x, wires=(0, 1))
            mat = op.fock_matrix(dims)
            return hl.math.expectation_value(n, mat @ state).real

        x = jnp.array([0.123, 0])
        grad_fn = hl.math.grad(f)

        grad = grad_fn(x)
        assert grad[0] > 0

        # With x[0] = 0, the phase should not affect anything, and x[0] = 0 is a local minimum
        x = jnp.zeros_like(x)
        grad = grad_fn(x)
        assert grad == pytest.approx(0)


@pytest.mark.unit
class TestAntiJaynesCummings:
    def test_init(self):
        op = hl.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "AntiJaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.AntiJaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.AntiJaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.AntiJaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

        # AJC is not periodic w.r.t. theta
        op = hl.AJC(5 * math.pi, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert simplified_op.parameters == pytest.approx(op.parameters)

    def test_label(self):
        op = hl.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.label() == "AJC"

    def test_fock_matrix_unitary(self):
        op = hl.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hl.math.eye(8)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    def test_fock_matrix_action(self):
        # Applying this gate to the vacuum state should give us 2 quanta,
        #   AJC(pi/2, 0)|0,0> ~= |1,1>
        dim = 4
        op = hl.AJC(math.pi / 2, 0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: dim})

        qumode_state = hl.math.zeros(dim)
        qumode_state[0] = 1
        qubit_state = hl.math.asarray([1, 0])
        state = hl.math.kron(qubit_state, qumode_state)
        evolved_state = matrix @ state

        expected_qumode_state = hl.math.zeros_like(qumode_state)
        expected_qumode_state[1] = 1
        expected_qubit_state = hl.math.asarray([0, 1])
        expected_state = hl.math.kron(expected_qubit_state, expected_qumode_state)

        assert hl.math.fidelity_statevector(evolved_state, expected_state) == pytest.approx(1)

        # Evolve again to obtain the original state
        evolved_state = matrix @ evolved_state
        assert hl.math.fidelity_statevector(evolved_state, state) == pytest.approx(1)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hl.AntiJaynesCummings(theta, phi, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hl.math.eye(8, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.AJC(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, 0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 2, 1: 4}
        # |0,0>
        state = hl.math.kron(jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0]))
        n = hl.N(1).fock_matrix(wire_dims=dims, wire_order=(0, 1))
        z = hl.math.expand_matrix(qp.Z(0).matrix(), (0,), wire_order=(0, 1), wire_dims=dims)
        z = hl.math.asarray(z, like=state)
        obs = n - z

        def f(x):
            op = hl.AJC(*x, wires=(0, 1))
            mat = op.fock_matrix(dims)
            return hl.math.expectation_value(obs, mat @ state).real

        x = jnp.array([0.123, 0])
        grad_fn = hl.math.grad(f)

        # Applying to |0,0> creates quanta in both, so we have positive gradient
        grad = grad_fn(x)
        assert grad[0] > 0

        # With x[0] = 0, the phase should not affect anything, and x[0] = 0 is a local minimum
        x = jnp.zeros_like(x)
        grad = grad_fn(x)
        assert grad == pytest.approx(0)


@pytest.mark.unit
class TestRabi:
    def test_init(self):
        op = hl.Rabi(0.5, 0.3, wires=[0, 1])
        assert op.name == "Rabi"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.Rabi(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.Rabi)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.Rabi(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.Rabi)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.Rabi(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.Rabi(0.5, 0.3, wires=[0, 1])
        assert op.label() == "RB"

    def test_fock_matrix_zero(self):
        op = hl.Rabi(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        assert matrix == pytest.approx(hl.math.eye(8), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hl.Rabi(0.5, 0.3, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hl.math.eye(8)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        r = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hl.Rabi(r, phi, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hl.math.eye(8, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.Rabi(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, 0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hl.Rabi(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4}).real  # ty:ignore[unresolved-attribute]

        x = jnp.array([0.123, 0])
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalDisplacement:
    def test_init(self):
        op = hl.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.ConditionalDisplacement)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.ConditionalDisplacement)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.ConditionalDisplacement(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.label() == "CD"

    def test_fock_matrix(self):
        dims = {0: 2, 1: 6}
        op = hl.ConditionalDisplacement(0.3, 0.5, wires=[0, 1])
        d = hl.D(0.3, 0.5, 1).fock_matrix(dims)
        dd = hl.math.dag(d)
        matrix = op.fock_matrix(dims)
        assert matrix == pytest.approx(hl.math.block_diag([d, dd]))

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        a = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hl.ConditionalDisplacement(a, phi, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hl.math.eye(12, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.CD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 2, 1: 4}

        def f(x):
            op = hl.CD(*x, wires=(0, 1))
            mat = op.fock_matrix(dims)
            # extract coherent state |-alpha>
            state = mat.reshape((2, 4, 2, 4))[1, :, 1, 0]  # ty:ignore[unresolved-attribute]
            x = hl.X(1).fock_matrix(dims)
            x = hl.math.asarray(x, like=state)
            return hl.math.expectation_value(x, state).real

        x = jnp.array([0.123, 0])
        grad_fn = hl.math.grad(f)
        grad = grad_fn(x)

        # Increasing displacement (r) should lower <x>, but phi=0 is a local maximum
        assert grad[0] < 0
        assert grad[1] == pytest.approx(0)

        # Now displace orthogonal to x, making r have no effect. Decreasing phi back
        # towards 0 should lower the value, meaning the grad > 0
        x = jnp.array([0.123, jnp.pi / 2])
        grad = grad_fn(x)
        assert grad[0] == pytest.approx(0)
        assert grad[1] > 0


@pytest.mark.unit
class TestConditionalXDisplacement:
    def test_init(self):
        op = hl.ConditionalXDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalXDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.ConditionalXDisplacement(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.ConditionalXDisplacement)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.ConditionalXDisplacement(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.ConditionalXDisplacement)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.ConditionalXDisplacement(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.ConditionalXDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.label() == "xCD"

    def test_fock_matrix_zero(self):
        op = hl.ConditionalXDisplacement(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        assert matrix == pytest.approx(hl.math.eye(8), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hl.ConditionalXDisplacement(0.3, 0.5, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hl.math.eye(12)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        a = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hl.ConditionalXDisplacement(a, phi, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hl.math.eye(12, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.XCD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hl.XCD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4}).real  # ty:ignore[unresolved-attribute]

        x = jnp.array([0.123, -0.456])
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalYDisplacement:
    def test_init(self):
        op = hl.ConditionalYDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalYDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.ConditionalYDisplacement(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.ConditionalYDisplacement)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.ConditionalYDisplacement(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.ConditionalYDisplacement)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.ConditionalYDisplacement(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.ConditionalYDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.label() == "yCD"

    def test_fock_matrix_zero(self):
        op = hl.ConditionalYDisplacement(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        assert matrix == pytest.approx(hl.math.eye(8), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hl.ConditionalYDisplacement(0.3, 0.5, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hl.math.eye(12)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        a = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hl.ConditionalYDisplacement(a, phi, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hl.math.eye(12, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.YCD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hl.YCD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4}).real  # ty:ignore[unresolved-attribute]

        x = jnp.array([0.123, -0.456])
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalSqueezing:
    def test_init(self):
        op = hl.ConditionalSqueezing(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalSqueezing"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.ConditionalSqueezing(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.ConditionalSqueezing)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.ConditionalSqueezing(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.ConditionalSqueezing)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.ConditionalSqueezing(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.ConditionalSqueezing(0.5, 0.3, wires=[0, 1])
        assert op.label() == "CS"

    def test_fock_matrix_zero(self):
        op = hl.ConditionalSqueezing(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        assert matrix == pytest.approx(hl.math.eye(8), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hl.ConditionalSqueezing(0.3, 0.5, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 8})
        eye = hl.math.eye(16)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        z = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hl.ConditionalSqueezing(z, phi, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 2, 1: 8})
        eye = hl.math.eye(16, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.CS(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def var(obs, state):
            return (
                hl.math.expectation_value(obs @ obs, state)
                - hl.math.expectation_value(obs, state) ** 2
            ).real

        dims = {0: 2, 1: 4}

        def f(x):
            obs = hl.X(1).fock_matrix(dims)
            obs = hl.math.asarray(obs, like="jax")
            mat = hl.CS(*x, wires=(0, 1)).fock_matrix(dims)
            # CS|1,0>
            state = mat.reshape((2, 4, 2, 4))[1, :, 1, 0]  # ty:ignore[unresolved-attribute]
            return var(obs, state)

        x = jnp.array([0.123, 0])
        grad_fn = hl.math.grad(f)
        grad = grad_fn(x)
        assert grad[0] > 0
        assert grad[1] == pytest.approx(0)


@pytest.mark.unit
class TestEchoedConditionalDisplacement:
    def test_init(self):
        op = hl.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        assert op.name == "EchoedConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        adj_op = op.adjoint()
        assert adj_op.parameters == [0.5, 0]

    def test_pow(self):
        op = hl.EchoedConditionalDisplacement(0.5, 0.123, wires=[0, 1])
        pow_op = op.pow(4)[0]
        assert pow_op == qp.Identity(pow_op.wires)

        pow_op = op.pow(3)[0]
        assert isinstance(pow_op, hl.EchoedConditionalDisplacement)
        assert pow_op.parameters == [0.5, 0.123]

    def test_simplify(self):
        op = hl.EchoedConditionalDisplacement(0, 0.123, wires=[0, 1])
        simplified_op = op.simplify()
        assert simplified_op == qp.X(0)

        op = hl.ECD(0.456, 0.123 + 2 * math.pi, wires=[0, 1])
        simplified_op = op.simplify()
        assert simplified_op.parameters == pytest.approx((0.456, 0.123))

    def test_label(self):
        op = hl.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        assert op.label() == "ECD"

    @pytest.mark.all_interfaces
    def test_fock_matrix_zero(self, like):
        # CD(0) = I, so ECD(0) = X @ I = X ⊗ I_{qumode}.
        params = hl.math.zeros(2, like=like)
        op = hl.EchoedConditionalDisplacement(*params, wires=[0, 1])
        dim = 4
        matrix = op.fock_matrix({0: 2, 1: dim})
        # X ⊗ I_{dim}
        x_mat = qp.X.compute_matrix()
        eye_dim = hl.math.eye(dim, like=like)
        expected = hl.math.kron(x_mat, eye_dim)
        assert matrix == pytest.approx(expected, abs=1e-6)

    @pytest.mark.all_interfaces
    def test_fock_matrix_unitary(self, like):
        params = hl.math.asarray([0.3, 0.5], like=like)
        op = hl.EchoedConditionalDisplacement(*params, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hl.math.eye(12, like=like)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.all_interfaces
    def test_fock_matrix_involutory(self, like):
        params = hl.math.asarray([0.3, 0.5], like=like)
        op = hl.EchoedConditionalDisplacement(*params, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        assert hl.math.dag(matrix) == pytest.approx(matrix, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.ECD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hl.ECD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4}).real  # ty:ignore[unresolved-attribute]

        x = jnp.array([0.123, -0.456])
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))

    def test_cd_decomp(self):
        tape = QuantumScript([hl.ECD(0.4, 0.123, wires=[0, 1])])
        [new_tape], _ = qp.decompose(tape, gate_set={hl.CD, qp.X})
        expected_tape = QuantumScript([
            hl.CD(0.2, 0.123, wires=(0, 1)),
            qp.X(0)
        ])
        check_tapes_approx(new_tape, expected_tape)

    def test_adjoint_decomp(self):
        tape = QuantumScript([qp.adjoint(hl.ECD(0.4, 0.123, wires=[0, 1]))])
        [new_tape], _ = qp.decompose(tape, gate_set={hl.ECD})
        expected_tape = QuantumScript([
            hl.ECD(0.4, 0.123, wires=(0, 1))
        ])
        check_tapes_approx(new_tape, expected_tape)

    def test_pow_involutory_decomp(self):
        tape = QuantumScript([
            qp.pow(hl.ECD(0.4, 0.123, wires=[0, 1]), 2)
        ])
        [new_tape], _ = qp.decompose(tape)
        assert new_tape.operations == []

def check_tapes_approx(tape1, tape2):
    """Check if two tapes are approximately equal."""
    for op1, op2 in zip(tape1.operations, tape2.operations, strict=True):
        assert type(op1) is type(op2)
        assert op1.wires == op2.wires
        for p1, p2 in zip(op1.parameters, op2.parameters, strict=True):
            assert hl.math.allclose(p1, p2)
