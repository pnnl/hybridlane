# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
class TestConditionalRotation:
    def test_init(self):
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        assert op.name == "ConditionalRotation"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalRotation)
        assert adj_op.parameters[0] == -0.5

    def test_pow(self):
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalRotation)
        assert pow_op[0].parameters[0] == 1.0

    def test_simplify(self):
        op = hqml.ConditionalRotation(0, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

        op = hqml.ConditionalRotation(1e-9, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        assert op.label() == "CR"

    def test_fock_matrix(self):
        theta = math.pi / 2
        op = hqml.ConditionalRotation(theta, wires=[0, 1])
        dim = 3
        matrix = op.fock_matrix({0: 2, 1: dim})
        r = hqml.Rotation.compute_fock_matrix((dim,), theta / 2)
        rd = hqml.math.dag(r)
        expected = hqml.math.block_diag([r, rd])
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(math.pi / 2)
        op = hqml.ConditionalRotation(theta, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hqml.math.eye(8, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(theta):
            op = hqml.CR(theta, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        theta = jnp.array(math.pi / 4)
        f(theta)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dim = 4

        # Should output |0>sin(xn/2) + |1>sin(-xn/2)
        def f(theta):
            op = hqml.CR(theta, wires=(0, 1))
            mat = op.fock_matrix({0: 2, 1: dim})
            return (mat @ jnp.ones(2 * dim)).imag.reshape(2, dim)

        theta = jnp.array(math.pi / 4)
        n = jnp.arange(dim)
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(theta)
        expected_grad = -n / 2 * jnp.cos(theta / 2 * n)
        assert grad[0] == pytest.approx(expected_grad)
        assert grad[1] == pytest.approx(-expected_grad)


@pytest.mark.unit
class TestSelectiveQubitRotation:
    def test_init(self):
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        assert op.name == "SelectiveQubitRotation"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.SelectiveQubitRotation)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.SelectiveQubitRotation)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.SelectiveQubitRotation(0, 0.3, 1, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.SQR(0.5, 0.3, 1, wires=[0, 1])
        assert op.label() == "SQR_{1}"

    def test_fock_matrix_unitary(self):
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hqml.math.eye(8)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hqml.SelectiveQubitRotation(theta, phi, 0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hqml.math.eye(8, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.SQR(*x, 1, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, 0])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 2, 1: 4}
        # |1,1>
        state = hqml.math.kron(jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0, 0.0, 0.0]))
        z = hqml.math.expand_matrix(
            qml.Z(0).matrix(), (0,), wire_order=(0, 1), wire_dims=dims
        )
        z = hqml.math.asarray(z, like=state)

        def f(x, n):
            op = hqml.SQR(*x, n, wires=(0, 1))
            mat = op.fock_matrix(dims)
            return hqml.math.expectation_value(z, mat @ state).real

        x = jnp.array([0.123, 0])
        grad_fn = hqml.math.grad(f)

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
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "JaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.JaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.JaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.JaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.label() == "JC"

    def test_fock_matrix_unitary(self):
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hqml.math.eye(8)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    def test_fock_matrix_action(self):
        # Test that we can move one quanta from qubit -> qumode
        #   JC(pi/2, 0)|1, 0> = |0, 1>
        dim = 4
        op = hqml.JaynesCummings(math.pi / 2, 0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: dim})

        qumode_state = hqml.math.zeros(dim)
        qumode_state[0] = 1
        qubit_state = hqml.math.asarray([0, 1])
        state = hqml.math.kron(qubit_state, qumode_state)
        evolved_state = matrix @ state

        expected_qumode_state = hqml.math.zeros_like(qumode_state)
        expected_qumode_state[1] = 1
        expected_qubit_state = hqml.math.asarray([1, 0])
        expected_state = hqml.math.kron(expected_qubit_state, expected_qumode_state)

        assert hqml.math.fidelity_statevector(
            evolved_state, expected_state
        ) == pytest.approx(1)

        # Evolve again to obtain the original state
        evolved_state = matrix @ evolved_state
        assert hqml.math.fidelity_statevector(evolved_state, state) == pytest.approx(1)

    def test_fock_matrix_commutes(self):
        # Check that the matrix conserves total excitation
        #   [JC, n + |1><1|] = 0
        dim = 5
        wire_dims = {0: 2, 1: dim}
        p1 = hqml.math.asarray([[0, 0], [0, 1]])
        p1 = hqml.math.expand_matrix(p1, (0,), wire_dims=wire_dims, wire_order=(0, 1))
        n = hqml.N(1).fock_matrix(wire_dims, wire_order=(0, 1))
        obs = p1 + n

        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        matrix = op.fock_matrix(wire_dims)
        comm = obs @ matrix - matrix @ obs
        assert comm == pytest.approx(0)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hqml.JaynesCummings(theta, phi, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hqml.math.eye(8, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.JC(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, 0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 2, 1: 4}
        # |1,0>
        state = hqml.math.kron(jnp.array([0.0, 1.0]), jnp.array([1.0, 0.0, 0.0, 0.0]))
        n = hqml.N(1).fock_matrix(wire_dims=dims, wire_order=(0, 1))

        def f(x):
            op = hqml.JC(*x, wires=(0, 1))
            mat = op.fock_matrix(dims)
            return hqml.math.expectation_value(n, mat @ state).real

        x = jnp.array([0.123, 0])
        grad_fn = hqml.math.grad(f)

        grad = grad_fn(x)
        assert grad[0] > 0

        # With x[0] = 0, the phase should not affect anything, and x[0] = 0 is a local minimum
        x = jnp.zeros_like(x)
        grad = grad_fn(x)
        assert grad == pytest.approx(0)


@pytest.mark.unit
class TestAntiJaynesCummings:
    def test_init(self):
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "AntiJaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.AntiJaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.AntiJaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.AntiJaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.label() == "AJC"

    def test_fock_matrix_unitary(self):
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hqml.math.eye(8)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    def test_fock_matrix_action(self):
        # Applying this gate to the vacuum state should give us 2 quanta,
        #   AJC(pi/2, 0)|0,0> ~= |1,1>
        dim = 4
        op = hqml.AJC(math.pi / 2, 0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: dim})

        qumode_state = hqml.math.zeros(dim)
        qumode_state[0] = 1
        qubit_state = hqml.math.asarray([1, 0])
        state = hqml.math.kron(qubit_state, qumode_state)
        evolved_state = matrix @ state

        expected_qumode_state = hqml.math.zeros_like(qumode_state)
        expected_qumode_state[1] = 1
        expected_qubit_state = hqml.math.asarray([0, 1])
        expected_state = hqml.math.kron(expected_qubit_state, expected_qumode_state)

        assert hqml.math.fidelity_statevector(
            evolved_state, expected_state
        ) == pytest.approx(1)

        # Evolve again to obtain the original state
        evolved_state = matrix @ evolved_state
        assert hqml.math.fidelity_statevector(evolved_state, state) == pytest.approx(1)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hqml.AntiJaynesCummings(theta, phi, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hqml.math.eye(8, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.AJC(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, 0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 2, 1: 4}
        # |0,0>
        state = hqml.math.kron(jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0, 0.0, 0.0]))
        n = hqml.N(1).fock_matrix(wire_dims=dims, wire_order=(0, 1))
        z = hqml.math.expand_matrix(
            qml.Z(0).matrix(), (0,), wire_order=(0, 1), wire_dims=dims
        )
        z = hqml.math.asarray(z, like=state)
        obs = n - z

        def f(x):
            op = hqml.AJC(*x, wires=(0, 1))
            mat = op.fock_matrix(dims)
            return hqml.math.expectation_value(obs, mat @ state).real

        x = jnp.array([0.123, 0])
        grad_fn = hqml.math.grad(f)

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
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        assert op.name == "Rabi"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.Rabi)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.Rabi)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.Rabi(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        assert op.label() == "RB"

    def test_fock_matrix_zero(self):
        op = hqml.Rabi(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        assert matrix == pytest.approx(hqml.math.eye(8), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hqml.math.eye(8)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        r = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hqml.Rabi(r, phi, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        eye = hqml.math.eye(8, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.Rabi(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, 0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hqml.Rabi(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4}).real

        x = jnp.array([0.123, 0])
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalDisplacement:
    def test_init(self):
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalDisplacement)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalDisplacement)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.ConditionalDisplacement(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.label() == "CD"

    def test_fock_matrix(self):
        dims = {0: 2, 1: 6}
        op = hqml.ConditionalDisplacement(0.3, 0.5, wires=[0, 1])
        d = hqml.D(0.3, 0.5, 1).fock_matrix(dims)
        dd = hqml.math.dag(d)
        matrix = op.fock_matrix(dims)
        assert matrix == pytest.approx(hqml.math.block_diag([d, dd]))

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        a = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hqml.ConditionalDisplacement(a, phi, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hqml.math.eye(12, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.CD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 2, 1: 4}

        def f(x):
            op = hqml.CD(*x, wires=(0, 1))
            mat = op.fock_matrix(dims)
            # extract coherent state |-alpha>
            state = mat.reshape((2, 4, 2, 4))[1, :, 1, 0]
            x = hqml.X(1).fock_matrix(dims)
            x = hqml.math.asarray(x, like=state)
            return hqml.math.expectation_value(x, state).real

        x = jnp.array([0.123, 0])
        grad_fn = hqml.math.grad(f)
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
        op = hqml.ConditionalXDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalXDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.ConditionalXDisplacement(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalXDisplacement)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.ConditionalXDisplacement(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalXDisplacement)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.ConditionalXDisplacement(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.ConditionalXDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.label() == "xCD"

    def test_fock_matrix_zero(self):
        op = hqml.ConditionalXDisplacement(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        assert matrix == pytest.approx(hqml.math.eye(8), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hqml.ConditionalXDisplacement(0.3, 0.5, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hqml.math.eye(12)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        a = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hqml.ConditionalXDisplacement(a, phi, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hqml.math.eye(12, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.XCD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hqml.XCD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4}).real

        x = jnp.array([0.123, -0.456])
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalYDisplacement:
    def test_init(self):
        op = hqml.ConditionalYDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalYDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.ConditionalYDisplacement(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalYDisplacement)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.ConditionalYDisplacement(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalYDisplacement)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.ConditionalYDisplacement(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.ConditionalYDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.label() == "yCD"

    def test_fock_matrix_zero(self):
        op = hqml.ConditionalYDisplacement(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        assert matrix == pytest.approx(hqml.math.eye(8), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hqml.ConditionalYDisplacement(0.3, 0.5, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hqml.math.eye(12)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        a = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hqml.ConditionalYDisplacement(a, phi, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hqml.math.eye(12, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.YCD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hqml.YCD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4}).real

        x = jnp.array([0.123, -0.456])
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalSqueezing:
    def test_init(self):
        op = hqml.ConditionalSqueezing(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalSqueezing"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.ConditionalSqueezing(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalSqueezing)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.ConditionalSqueezing(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalSqueezing)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.ConditionalSqueezing(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.ConditionalSqueezing(0.5, 0.3, wires=[0, 1])
        assert op.label() == "CS"

    def test_fock_matrix_zero(self):
        op = hqml.ConditionalSqueezing(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 4})
        assert matrix == pytest.approx(hqml.math.eye(8), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hqml.ConditionalSqueezing(0.3, 0.5, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 8})
        eye = hqml.math.eye(16)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        z = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hqml.ConditionalSqueezing(z, phi, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 8})
        eye = hqml.math.eye(16, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.CS(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def var(obs, state):
            return (
                hqml.math.expectation_value(obs @ obs, state)
                - hqml.math.expectation_value(obs, state) ** 2
            ).real

        dims = {0: 2, 1: 4}

        def f(x):
            obs = hqml.X(1).fock_matrix(dims)
            obs = hqml.math.asarray(obs, like="jax")
            mat = hqml.CS(*x, wires=(0, 1)).fock_matrix(dims)
            # CS|1,0>
            state = mat.reshape((2, 4, 2, 4))[1, :, 1, 0]
            return var(obs, state)

        x = jnp.array([0.123, 0])
        grad_fn = hqml.math.grad(f)
        grad = grad_fn(x)
        assert grad[0] > 0
        assert grad[1] == pytest.approx(0)


@pytest.mark.unit
class TestEchoedConditionalDisplacement:
    def test_init(self):
        op = hqml.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        assert op.name == "EchoedConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        adj_op = op.adjoint()[0]
        assert adj_op.parameters == [-0.5, 0]

    def test_pow(self):
        op = hqml.EchoedConditionalDisplacement(0.5, 0.123, wires=[0, 1])
        pow_op = op.pow(4)[0]
        assert pow_op.parameters == [2, 0.123]

    def test_simplify(self):
        op = hqml.EchoedConditionalDisplacement(0, 0.123, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        assert op.label() == "ECD"

    def test_fock_matrix_zero(self):
        # CD(0) = I, so ECD(0) = X @ I = X ⊗ I_{qumode}.
        op = hqml.EchoedConditionalDisplacement(0.0, 0.0, wires=[0, 1])
        dim = 4
        matrix = op.fock_matrix({0: 2, 1: dim})
        # X ⊗ I_{dim}
        x_mat = qml.X.compute_matrix()
        eye_dim = hqml.math.eye(dim)
        expected = hqml.math.kron(x_mat, eye_dim)
        assert matrix == pytest.approx(expected, abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hqml.EchoedConditionalDisplacement(0.3, 0.5, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hqml.math.eye(12)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        a = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hqml.EchoedConditionalDisplacement(a, phi, wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 6})
        eye = hqml.math.eye(12, like="jax")
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.ECD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hqml.ECD(*x, wires=(0, 1))
            return op.fock_matrix({0: 2, 1: 4}).real

        x = jnp.array([0.123, -0.456])
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))
