# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: N806
import math

import jax
import jax.numpy as jnp
import pennylane as qp
import pytest

import hybridlane as hl


@pytest.mark.unit
class TestSelectiveNumberArbitraryPhase:
    def test_init(self):
        op = hl.SelectiveNumberArbitraryPhase(0.5, 1, 1)
        assert op.name == "SelectiveNumberArbitraryPhase"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qp.wires.Wires([1])

    def test_adjoint(self):
        op = hl.SelectiveNumberArbitraryPhase(0.5, 1, 0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.SelectiveNumberArbitraryPhase)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        op = hl.SelectiveNumberArbitraryPhase(0.5, 1, 0)
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.SelectiveNumberArbitraryPhase)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        op = hl.SelectiveNumberArbitraryPhase(0, 1, 0)
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.SNAP(0.5, 1, 0)
        assert op.label() == "SNAP_{1}"

    def test_fock_matrix(self):
        op = hl.SNAP(0.5, 1, 0)
        matrix = op.fock_matrix({0: 4})
        expected = hl.math.diag([1, hl.math.exp(0.5j), 1, 1])
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        param = hl.math.asarray(0.5, like="jax")
        op = hl.SNAP(param, 1, 0)
        matrix = op.fock_matrix({0: 4})
        expected = hl.math.diag([1, hl.math.exp(0.5j), 1, 1], like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        @jax.jit
        def f(theta):
            op = hl.SNAP(theta, 3, wires=0)
            return op.fock_matrix({0: 4})

        theta = jnp.array(0.5)
        f(theta)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        def f(theta):
            op = hl.SNAP(theta, 3, wires=0)
            return hl.math.real(op.fock_matrix({0: 4}))

        theta = jnp.array(0.5)
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(theta)
        assert not hl.math.any(hl.math.isnan(grad))


@pytest.mark.unit
class TestDisplacement:
    def test_init(self):
        op = hl.Displacement(0.5, 0.3, wires=0)
        assert op.name == "Displacement"
        assert op.num_params == 2
        assert op.num_wires == 1
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires(0)

    def test_adjoint(self):
        op = hl.Displacement(0.5, 0.3, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.Displacement)
        assert adj_op.parameters[0] == 0.5
        assert adj_op.parameters[1] == pytest.approx(0.3 + math.pi)

    def test_label(self):
        op = hl.Displacement(0.5, 0.3, wires=0)
        assert op.label() == "D"

    def test_fock_matrix_action(self):
        from scipy.special import factorial

        alpha_r, alpha_phi = 0.3, 0.5
        alpha = alpha_r * hl.math.exp(1j * alpha_phi)
        dim = 10

        d_mat = hl.D.compute_fock_matrix((dim,), alpha_r, alpha_phi)
        d_on_vacuum = d_mat[:, 0]  # D|0>  # ty:ignore[invalid-argument-type, not-subscriptable]

        n = hl.math.arange(dim)
        expected = hl.math.exp(-(abs(alpha) ** 2) / 2) * alpha**n / hl.math.sqrt(factorial(n))
        assert d_on_vacuum == pytest.approx(expected, abs=1e-8)

    def test_fock_matrix_unitary(self):
        dim = 16
        op = hl.Displacement(0.3, 0.5, wires=0)
        matrix = op.fock_matrix({0: dim})
        assert matrix @ hl.math.dag(matrix) == pytest.approx(hl.math.eye(dim), abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        from scipy.special import factorial

        alpha_r = jnp.array(0.3)
        alpha_phi = jnp.array(0.5)
        alpha = alpha_r * hl.math.exp(1j * alpha_phi)
        dim = 10

        op = hl.Displacement(alpha_r, alpha_phi, wires=0)  # ty:ignore[invalid-argument-type]
        d_on_vacuum = op.fock_matrix({0: dim})[:, 0]  # ty:ignore[invalid-argument-type, not-subscriptable]

        n = jnp.arange(dim)
        expected = (
            hl.math.exp(-(abs(alpha) ** 2) / 2) * alpha**n / hl.math.sqrt(jnp.array(factorial(n)))
        )
        assert hl.math.get_interface(d_on_vacuum) == "jax"
        assert d_on_vacuum == pytest.approx(expected, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        @jax.jit
        def f(x):
            op = hl.D(*x, wires=0)
            return op.fock_matrix({0: 4})

        x = jnp.array([0.5, 0.123])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        dims = {0: 4}

        def f(x):
            op = hl.D(*x, wires=0)
            mat = op.fock_matrix(dims)
            state = mat[:, 0]  # ty:ignore[invalid-argument-type, not-subscriptable]
            x = hl.X(0).fock_matrix(dims)
            x = hl.math.asarray(x, like=state)
            return hl.math.expectation_value(x, state).real

        x = jnp.array([0.123, 0])
        grad_fn = hl.math.grad(f)
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

    @pytest.mark.all_interfaces
    def test_heisenberg_rep(self, like):
        a = hl.math.asarray(0.5, like=like)
        phi = hl.math.asarray(0.3, like=like)
        M = hl.Displacement._heisenberg_rep([a, phi])
        assert hl.math.is_symplectic(M)
        assert hl.math.get_interface(M) == like
        assert hl.math.get_dtype_name(M) == "float64"

        expected = hl.math.asarray(
            [
                [1, 0, 0],
                [math.sqrt(2) * math.cos(0.3) * 0.5, 1, 0],
                [math.sqrt(2) * math.sin(0.3) * 0.5, 0, 1],
            ]
        )
        assert pytest.approx(expected, abs=1e-6) == M

    @pytest.mark.jax
    def test_heisenberg_rep_jit(self):
        @jax.jit
        def f(x):
            return hl.Displacement._heisenberg_rep(x)

        x = jnp.array([0.5, 0.3])
        f(x)  # errors if jit fails


@pytest.mark.unit
class TestRotation:
    def test_init(self):
        op = hl.Rotation(0.5, wires=0)
        assert op.name == "Rotation"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.wires == qp.wires.Wires(0)

    def test_adjoint(self):
        op = hl.Rotation(0.5, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.Rotation)
        assert adj_op.parameters[0] == -0.5

    def test_simplify(self):
        op = hl.Rotation(0, wires=0)
        assert isinstance(op.simplify(), qp.Identity)

        op = hl.Rotation(1e-9, wires=0)
        assert isinstance(op.simplify(), qp.Identity)

    def test_label(self):
        op = hl.Rotation(0.5, wires=0)
        assert op.label() == "R"

    def test_fock_matrix_zero(self):
        op = hl.Rotation(0.0, wires=0)
        matrix = op.fock_matrix({0: 4})
        assert matrix == pytest.approx(hl.math.eye(4))

    def test_fock_matrix_half_pi(self):
        op = hl.Rotation(math.pi / 2, wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hl.math.diag([1, -1j, -1, 1j])
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        theta = jnp.array(math.pi / 2)
        op = hl.Rotation(theta, wires=0)  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 4})
        expected = hl.math.diag(jnp.array([1, -1j, -1, 1j]))
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        @jax.jit
        def f(theta):
            op = hl.R(theta, 0)
            return op.fock_matrix({0: 4})

        theta = jnp.array(math.pi / 4)
        f(theta)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        dim = 4

        # Should output cos(xn)
        def f(theta):
            op = hl.R(theta, 0)
            mat = op.fock_matrix({0: dim})
            return (mat @ jnp.ones(dim)).real  # ty:ignore[unsupported-operator]

        theta = jnp.array(math.pi / 4)
        n = jnp.arange(dim)
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(theta)
        expected_grad = -n * jnp.sin(theta * n)
        assert grad == pytest.approx(expected_grad)

    @pytest.mark.all_interfaces
    def test_heisenberg_rep(self, like):
        theta = hl.math.asarray(0.5, like=like)
        M = hl.Rotation._heisenberg_rep([theta])

        assert hl.math.is_symplectic(M)
        assert hl.math.get_interface(M) == like
        assert hl.math.get_dtype_name(M) == "float64"

        expected = hl.math.symplectic.rotation(theta)
        assert pytest.approx(expected, abs=1e-6) == M

    @pytest.mark.jax
    def test_heisenberg_rep_jit(self):
        @jax.jit
        def f(theta):
            return hl.Rotation._heisenberg_rep([theta])

        f(jnp.array(0.5))  # errors if jit fails


@pytest.mark.unit
class TestSqueezing:
    def test_init(self):
        op = hl.Squeezing(0.5, 0.3, wires=0)
        assert op.name == "Squeezing"
        assert op.num_params == 2
        assert op.num_wires == 1
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires(0)

    def test_adjoint(self):
        op = hl.Squeezing(0.5, 0.3, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.Squeezing)
        assert adj_op.parameters[0] == -0.5
        assert adj_op.parameters[1] == 0.3

    def test_simplify(self):
        op = hl.S(0.5, 0.3 + math.pi, wires=0)
        expected_op = hl.S(0.5, 0.3, wires=0)
        assert hl.math.allclose(op.simplify().parameters, expected_op.parameters)

    def test_label(self):
        op = hl.Squeezing(0.5, 0.3, wires=0)
        assert op.label() == "S"

    def test_fock_matrix_zero(self):
        op = hl.Squeezing(0.0, 0.0, wires=0)
        matrix = op.fock_matrix({0: 4})
        assert matrix == pytest.approx(hl.math.eye(4), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hl.Squeezing(0.3, 0.5, wires=0)
        matrix = op.fock_matrix({0: 8})
        eye = hl.math.eye(8)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    def test_fock_matrix_periodic(self):
        # comes from the discussion on p. 33
        op = hl.Squeezing(0.3, 0.5, wires=0)
        matrix = op.fock_matrix({0: 8})

        op2 = hl.Squeezing(0.3, 0.5 + math.pi, wires=0)
        matrix2 = op2.fock_matrix({0: 8})
        assert matrix2 == pytest.approx(matrix)

    def test_fock_matrix_action(self):
        # Check that the uncertainty in x decreases at the expense of p
        def var(obs, state):
            return (
                hl.math.expectation_value(obs @ obs, state)
                - hl.math.expectation_value(obs, state) ** 2
            )

        dim = 8
        obs = hl.X.compute_fock_matrix((dim,))
        vacuum = hl.math.eye(dim)[:, 0]
        mat = hl.S.compute_fock_matrix((dim,), 1, 0)
        state = mat[:, 0]  # S|0>  # ty:ignore[invalid-argument-type, not-subscriptable]
        var_x = var(obs, state)
        assert var_x < var(obs, vacuum)

        obs = hl.P.compute_fock_matrix((dim,))
        var_p = var(obs, state)
        assert var_p > var(obs, vacuum)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        r = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hl.Squeezing(r, phi, wires=0)
        matrix = op.fock_matrix({0: 8})
        eye = hl.math.eye(8, like="jax")
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        @jax.jit
        def f(x):
            op = hl.S(*x, wires=0)
            return op.fock_matrix({0: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        def var(obs, state):
            return (
                hl.math.expectation_value(obs @ obs, state)
                - hl.math.expectation_value(obs, state) ** 2
            ).real

        dim = 4

        def f(x):
            obs = hl.X.compute_fock_matrix((dim,))
            obs = hl.math.asarray(obs, like="jax")
            mat = hl.S.compute_fock_matrix((dim,), *x)
            state = mat[:, 0]  # S|0>  # ty:ignore[invalid-argument-type, not-subscriptable]
            return var(obs, state)

        # Evaluating Var(x) results in:
        #  - First parameter squeezes more, reducing variance -> negative gradient
        #  - Second parameter controls alignment, at 0 is a local minimum
        x = jnp.array([0.123, 0])
        grad_fn = hl.math.grad(f)
        grad = grad_fn(x)
        assert grad[0] < 0
        assert grad[1] == pytest.approx(0)

    @pytest.mark.all_interfaces
    def test_heisenberg_rep(self, like):
        r = hl.math.asarray(0.3, like=like)
        theta = hl.math.asarray(0, like=like)
        M = hl.Squeezing._heisenberg_rep([r, theta])
        assert hl.math.is_symplectic(M)
        assert hl.math.get_interface(M) == like
        assert hl.math.get_dtype_name(M) == "float64"

        # Check its form in the "mode" basis, eq. 167 of liu2026hybrid.
        c, s = hl.math.cosh(r), hl.math.sinh(r)
        expected_fock = hl.math.asarray(
            [
                [1, 0, 0],
                [0, c, -s],
                [0, -s, c],
            ]
        )
        assert hl.math.to_fock_space(M) == pytest.approx(expected_fock)

        eigvals, vecs = hl.math.linalg.eig(M)
        indices = hl.math.argsort(eigvals)
        eigvals = eigvals[indices]
        assert hl.math.allclose(eigvals, [hl.math.exp(-r), 1, hl.math.exp(r)])

        # Now check with a rotation
        theta = hl.math.asarray(0.123, like=like)
        M = hl.Squeezing._heisenberg_rep([r, theta])
        assert hl.math.is_symplectic(M)
        assert hl.math.get_interface(M) == like
        assert hl.math.get_dtype_name(M) == "float64"

        eigvals, vecs = hl.math.linalg.eig(M)
        indices = hl.math.argsort(eigvals)
        eigvals = eigvals[indices]
        vecs = vecs[:, indices]
        assert hl.math.allclose(eigvals, [hl.math.exp(-r), 1, hl.math.exp(r)])

        # Constant vector should have eigenvalue 1
        assert eigvals[1] == pytest.approx(1)
        assert hl.math.allclose(vecs[..., 1], [1, 0, 0])

        # The vector with eigenvalue exp(-r) should be a slightly rotated version of x,
        # eq. 161
        rotated_x = [0, hl.math.cos(theta), hl.math.sin(theta)]
        assert hl.math.allclose(vecs[..., 0], rotated_x)

        # Remaining vector with eigenvalue exp(r) is a rotated version of p
        # eq. 162
        rotated_p = [0, -hl.math.sin(theta), hl.math.cos(theta)]
        assert hl.math.allclose(vecs[..., 2], rotated_p)

    @pytest.mark.jax
    def test_heisenberg_rep_jit(self):
        @jax.jit
        def f(x):
            return hl.Squeezing._heisenberg_rep(x)

        f(jnp.array([0.3, 0.5]))  # errors if jit fails


@pytest.mark.unit
class TestKerr:
    def test_init(self):
        op = hl.Kerr(0.5, wires=0)
        assert op.name == "Kerr"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.wires == qp.wires.Wires(0)

    def test_adjoint(self):
        op = hl.Kerr(0.5, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.Kerr)
        assert adj_op.parameters[0] == -0.5

    def test_simplify(self):
        op = hl.Kerr(0, wires=0)
        assert isinstance(op.simplify(), qp.Identity)

        op = hl.Kerr(0.123 + 2 * math.pi, wires=0)
        assert op.simplify().parameters == pytest.approx([0.123])

    def test_label(self):
        op = hl.Kerr(0.5, wires=0)
        assert op.label() == "K"

    def test_fock_matrix(self):
        kappa = math.pi / 4
        op = hl.Kerr(kappa, wires=0)
        dim = 6
        matrix = op.fock_matrix({0: dim})
        n = hl.math.arange(dim)
        expected = hl.math.diag(hl.math.exp(-1j * kappa * n**2))
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    def test_fock_matrix_periodic(self):
        wire_dims = {0: 8}
        op = hl.K(0.123, wires=0)
        op2 = hl.K(0.123 + 2 * math.pi, wires=0)
        assert op.fock_matrix(wire_dims) == pytest.approx(op2.fock_matrix(wire_dims))

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        kappa = jnp.array(math.pi / 4)
        op = hl.Kerr(kappa, wires=0)  # ty:ignore[invalid-argument-type]
        dim = 6
        matrix = op.fock_matrix({0: dim})
        n = hl.math.arange(dim, like=kappa)
        expected = hl.math.diag(hl.math.exp(-1j * kappa * n**2))
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.K(*x, wires=0)
            return op.fock_matrix({0: 4})

        x = jnp.array([0.123])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        # Should output cos(xn^2)
        def f(x):
            op = hl.K(x, wires=0)
            mat = op.fock_matrix({0: 4})
            return (mat @ jnp.ones(4)).real  # ty:ignore[unsupported-operator]

        x = jnp.array(0.123)
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        n = jnp.arange(4)
        expected_grad = -jnp.sin(x * n**2) * n**2
        assert grad == pytest.approx(expected_grad)


@pytest.mark.unit
class TestCubicPhase:
    def test_init(self):
        op = hl.CubicPhase(0.5, wires=0)
        assert op.name == "CubicPhase"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.wires == qp.wires.Wires(0)

    def test_adjoint(self):
        op = hl.CubicPhase(0.5, wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.CubicPhase)
        assert adj_op.parameters[0] == -0.5

    def test_simplify(self):
        op = hl.CubicPhase(0, wires=0)
        assert isinstance(op.simplify(), qp.Identity)

        op = hl.CubicPhase(1e-9, wires=0)
        assert isinstance(op.simplify(), qp.Identity)

    def test_label(self):
        op = hl.CubicPhase(0.5, wires=0)
        assert op.label() == "C"

    def test_fock_matrix_zero(self):
        op = hl.CubicPhase(0.0, wires=0)
        matrix = op.fock_matrix({0: 4})
        assert matrix == pytest.approx(hl.math.eye(4), abs=1e-6)

    def test_fock_matrix_unitary(self):
        dim = 6
        op = hl.CubicPhase(0.1, wires=0)
        matrix = op.fock_matrix({0: dim})
        eye = hl.math.eye(dim)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        dim = 6
        r = jnp.array(0.1)
        op = hl.CubicPhase(r, wires=0)  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: dim})
        eye = hl.math.eye(dim, like=r)
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.C(*x, wires=0)
            return op.fock_matrix({0: 4})

        x = jnp.array([0.123])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        # todo: don't have a good understanding of this gate to build a solid test
        def f(x):
            op = hl.C(x, wires=0)
            return op.fock_matrix({0: 4}).real  # ty:ignore[unresolved-attribute]

        x = jnp.array(0.123)
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))
