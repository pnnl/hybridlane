# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
class TestSelectiveNumberArbitraryPhase:
    def test_init(self):
        op = hqml.SelectiveNumberArbitraryPhase(0.5, 1, 1)
        assert op.name == "SelectiveNumberArbitraryPhase"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qml.wires.Wires([1])

    def test_adjoint(self):
        op = hqml.SelectiveNumberArbitraryPhase(0.5, 1, 0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.SelectiveNumberArbitraryPhase)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        op = hqml.SelectiveNumberArbitraryPhase(0.5, 1, 0)
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.SelectiveNumberArbitraryPhase)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        op = hqml.SelectiveNumberArbitraryPhase(0, 1, 0)
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.SNAP(0.5, 1, 0)
        assert op.label() == "SNAP_{1}"

    def test_fock_matrix(self):
        op = hqml.SNAP(0.5, 1, 0)
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.diag([1, hqml.math.exp(0.5j), 1, 1])
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        param = hqml.math.asarray(0.5, like="jax")
        op = hqml.SNAP(param, 1, 0)
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.diag([1, hqml.math.exp(0.5j), 1, 1], like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(theta):
            op = hqml.SNAP(theta, 3, wires=0)
            return op.fock_matrix({0: 4})

        theta = jnp.array(0.5)
        f(theta)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(theta):
            op = hqml.SNAP(theta, 3, wires=0)
            return hqml.math.real(op.fock_matrix({0: 4}))

        theta = jnp.array(0.5)
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(theta)
        assert not hqml.math.any(hqml.math.isnan(grad))


@pytest.mark.unit
class TestDisplacement:
    def test_init(self):
        op = hqml.Displacement(0.5, 0.3, wires=0)
        assert op.name == "Displacement"
        assert op.num_params == 2
        assert op.num_wires == 1
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires(0)

    def test_adjoint(self):
        op = hqml.Displacement(0.5, 0.3, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.Displacement)
        assert adj_op.parameters[0] == 0.5
        assert adj_op.parameters[1] == pytest.approx(0.3 + math.pi)

    def test_label(self):
        op = hqml.Displacement(0.5, 0.3, wires=0)
        assert op.label() == "D"

    def test_fock_matrix_action(self):
        from scipy.special import factorial

        alpha_r, alpha_phi = 0.3, 0.5
        alpha = alpha_r * hqml.math.exp(1j * alpha_phi)
        dim = 10

        d_mat = hqml.D.compute_fock_matrix((dim,), alpha_r, alpha_phi)
        d_on_vacuum = d_mat[:, 0]  # D|0>

        n = hqml.math.arange(dim)
        expected = (
            hqml.math.exp(-(abs(alpha) ** 2) / 2)
            * alpha**n
            / hqml.math.sqrt(factorial(n))
        )
        assert d_on_vacuum == pytest.approx(expected, abs=1e-8)

    def test_fock_matrix_unitary(self):
        dim = 16
        op = hqml.Displacement(0.3, 0.5, wires=0)
        matrix = op.fock_matrix({0: dim})
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(
            hqml.math.eye(dim), abs=1e-6
        )

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp
        from scipy.special import factorial

        alpha_r = jnp.array(0.3)
        alpha_phi = jnp.array(0.5)
        alpha = alpha_r * hqml.math.exp(1j * alpha_phi)
        dim = 10

        op = hqml.Displacement(alpha_r, alpha_phi, wires=0)
        d_on_vacuum = op.fock_matrix({0: dim})[:, 0]

        n = jnp.arange(dim)
        expected = (
            hqml.math.exp(-(abs(alpha) ** 2) / 2)
            * alpha**n
            / hqml.math.sqrt(jnp.array(factorial(n)))
        )
        assert hqml.math.get_interface(d_on_vacuum) == "jax"
        assert d_on_vacuum == pytest.approx(expected, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.D(*x, wires=0)
            return op.fock_matrix({0: 4})

        x = jnp.array([0.5, 0.123])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 4}

        def f(x):
            op = hqml.D(*x, wires=0)
            mat = op.fock_matrix(dims)
            state = mat[:, 0]  # extract coherent state
            x = hqml.X(0).fock_matrix(dims)
            x = hqml.math.asarray(x, like=state)
            return hqml.math.expectation_value(x, state).real

        x = jnp.array([0.123, 0])
        grad_fn = hqml.math.grad(f)
        grad = grad_fn(x)

        # Increasing displacement (r) should boost <x>, but phi=0 is a local maximum
        assert grad[0] > 0
        assert grad[1] == pytest.approx(0)

        # Now displace orthogonal to x, making r have no effect. Decreasing phi back
        # towards 0 should boost our value, meaning the grad < 0
        x = jnp.array([0.123, jnp.pi / 2])
        grad = grad_fn(x)
        assert grad[0] == pytest.approx(0)
        assert grad[1] < 0


@pytest.mark.unit
class TestRotation:
    def test_init(self):
        op = hqml.Rotation(0.5, wires=0)
        assert op.name == "Rotation"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires(0)

    def test_adjoint(self):
        op = hqml.Rotation(0.5, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.Rotation)
        assert adj_op.parameters[0] == -0.5

    def test_simplify(self):
        op = hqml.Rotation(0, wires=0)
        assert isinstance(op.simplify(), qml.Identity)

        op = hqml.Rotation(1e-9, wires=0)
        assert isinstance(op.simplify(), qml.Identity)

    def test_label(self):
        op = hqml.Rotation(0.5, wires=0)
        assert op.label() == "R"

    def test_fock_matrix_zero(self):
        op = hqml.Rotation(0.0, wires=0)
        matrix = op.fock_matrix({0: 4})
        assert matrix == pytest.approx(hqml.math.eye(4))

    def test_fock_matrix_half_pi(self):
        op = hqml.Rotation(math.pi / 2, wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.diag([1, -1j, -1, 1j])
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(math.pi / 2)
        op = hqml.Rotation(theta, wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.diag(jnp.array([1, -1j, -1, 1j]))
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(theta):
            op = hqml.R(theta, 0)
            return op.fock_matrix({0: 4})

        theta = jnp.array(math.pi / 4)
        f(theta)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dim = 4

        # Should output cos(xn)
        def f(theta):
            op = hqml.R(theta, 0)
            mat = op.fock_matrix({0: dim})
            return (mat @ jnp.ones(dim)).real

        theta = jnp.array(math.pi / 4)
        n = jnp.arange(dim)
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(theta)
        expected_grad = -n * jnp.sin(theta * n)
        assert grad == pytest.approx(expected_grad)


@pytest.mark.unit
class TestSqueezing:
    def test_init(self):
        op = hqml.Squeezing(0.5, 0.3, wires=0)
        assert op.name == "Squeezing"
        assert op.num_params == 2
        assert op.num_wires == 1
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires(0)

    def test_adjoint(self):
        import numpy as np

        op = hqml.Squeezing(0.5, 0.3, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.Squeezing)
        assert adj_op.parameters[0] == 0.5
        assert adj_op.parameters[1] == pytest.approx((0.3 + np.pi) % (2 * np.pi))

    def test_label(self):
        op = hqml.Squeezing(0.5, 0.3, wires=0)
        assert op.label() == "S"

    def test_fock_matrix_zero(self):
        op = hqml.Squeezing(0.0, 0.0, wires=0)
        matrix = op.fock_matrix({0: 4})
        assert matrix == pytest.approx(hqml.math.eye(4), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hqml.Squeezing(0.3, 0.5, wires=0)
        matrix = op.fock_matrix({0: 8})
        eye = hqml.math.eye(8)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    def test_fock_matrix_action(self):
        # Check that the uncertainty in x decreases at the expense of p
        def var(obs, state):
            return (
                hqml.math.expectation_value(obs @ obs, state)
                - hqml.math.expectation_value(obs, state) ** 2
            )

        dim = 8
        obs = hqml.X.compute_fock_matrix((dim,))
        vacuum = hqml.math.eye(dim)[:, 0]
        mat = hqml.S.compute_fock_matrix((dim,), 1, 0)
        state = mat[:, 0]  # S|0>
        var_x = var(obs, state)
        assert var_x < var(obs, vacuum)

        obs = hqml.P.compute_fock_matrix((dim,))
        var_p = var(obs, state)
        assert var_p > var(obs, vacuum)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        r = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hqml.Squeezing(r, phi, wires=0)
        matrix = op.fock_matrix({0: 8})
        eye = hqml.math.eye(8, like="jax")
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.S(*x, wires=0)
            return op.fock_matrix({0: 4})

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

        dim = 4

        def f(x):
            obs = hqml.X.compute_fock_matrix((dim,))
            obs = hqml.math.asarray(obs, like="jax")
            mat = hqml.S.compute_fock_matrix((dim,), *x)
            state = mat[:, 0]  # S|0>
            return var(obs, state)

        # Evaluating Var(x) results in:
        #  - First parameter squeezes more, reducing variance -> negative gradient
        #  - Second parameter controls alignment, at 0 is a local minimum
        x = jnp.array([0.123, 0])
        grad_fn = hqml.math.grad(f)
        grad = grad_fn(x)
        assert grad[0] < 0
        assert grad[1] == pytest.approx(0)


@pytest.mark.unit
class TestKerr:
    def test_init(self):
        op = hqml.Kerr(0.5, wires=0)
        assert op.name == "Kerr"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires(0)

    def test_adjoint(self):
        op = hqml.Kerr(0.5, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.Kerr)
        assert adj_op.parameters[0] == -0.5

    def test_simplify(self):
        op = hqml.Kerr(0, wires=0)
        assert isinstance(op.simplify(), qml.Identity)

        op = hqml.Kerr(1e-9, wires=0)
        assert isinstance(op.simplify(), qml.Identity)

    def test_label(self):
        op = hqml.Kerr(0.5, wires=0)
        assert op.label() == "K"

    def test_fock_matrix(self):
        kappa = math.pi / 4
        op = hqml.Kerr(kappa, wires=0)
        dim = 6
        matrix = op.fock_matrix({0: dim})
        n = hqml.math.arange(dim)
        expected = hqml.math.diag(hqml.math.exp(-1j * kappa * n**2))
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        kappa = jnp.array(math.pi / 4)
        op = hqml.Kerr(kappa, wires=0)
        dim = 6
        matrix = op.fock_matrix({0: dim})
        n = hqml.math.arange(dim, like=kappa)
        expected = hqml.math.diag(hqml.math.exp(-1j * kappa * n**2))
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.K(*x, wires=0)
            return op.fock_matrix({0: 4})

        x = jnp.array([0.123])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        # Should output cos(xn^2)
        def f(x):
            op = hqml.K(x, wires=0)
            mat = op.fock_matrix({0: 4})
            return (mat @ jnp.ones(4)).real

        x = jnp.array(0.123)
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(x)
        n = jnp.arange(4)
        expected_grad = -jnp.sin(x * n**2) * n**2
        assert grad == pytest.approx(expected_grad)


@pytest.mark.unit
class TestCubicPhase:
    def test_init(self):
        op = hqml.CubicPhase(0.5, wires=0)
        assert op.name == "CubicPhase"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires(0)

    def test_adjoint(self):
        op = hqml.CubicPhase(0.5, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.CubicPhase)
        assert adj_op.parameters[0] == -0.5

    def test_simplify(self):
        op = hqml.CubicPhase(0, wires=0)
        assert isinstance(op.simplify(), qml.Identity)

        op = hqml.CubicPhase(1e-9, wires=0)
        assert isinstance(op.simplify(), qml.Identity)

    def test_label(self):
        op = hqml.CubicPhase(0.5, wires=0)
        assert op.label() == "C"

    def test_fock_matrix_zero(self):
        op = hqml.CubicPhase(0.0, wires=0)
        matrix = op.fock_matrix({0: 4})
        assert matrix == pytest.approx(hqml.math.eye(4), abs=1e-6)

    def test_fock_matrix_unitary(self):
        dim = 6
        op = hqml.CubicPhase(0.1, wires=0)
        matrix = op.fock_matrix({0: dim})
        eye = hqml.math.eye(dim)
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        dim = 6
        r = jnp.array(0.1)
        op = hqml.CubicPhase(r, wires=0)
        matrix = op.fock_matrix({0: dim})
        eye = hqml.math.eye(dim, like=r)
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.C(*x, wires=0)
            return op.fock_matrix({0: 4})

        x = jnp.array([0.123])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        # todo: don't have a good understanding of this gate to build a solid test
        def f(x):
            op = hqml.C(x, wires=0)
            return op.fock_matrix({0: 4}).real

        x = jnp.array(0.123)
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))
