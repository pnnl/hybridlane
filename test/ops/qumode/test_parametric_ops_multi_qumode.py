# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qp
import pytest

import hybridlane as hl


@pytest.mark.unit
class TestTwoModeSum:
    def test_init(self):
        op = hl.TwoModeSum(0.5, wires=[0, 1])
        assert op.name == "TwoModeSum"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.TwoModeSum(0.5, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.TwoModeSum)
        assert adj_op.parameters[0] == -0.5

    def test_pow(self):
        op = hl.TwoModeSum(0.5, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.TwoModeSum)
        assert pow_op[0].parameters[0] == 1.0

    def test_simplify(self):
        op = hl.TwoModeSum(0, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

        op = hl.TwoModeSum(1e-9, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.TwoModeSum(0.5, wires=[0, 1])
        assert op.label() == "SUM"

    def test_fock_matrix_zero(self):
        op = hl.TwoModeSum(0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 3, 1: 3})
        assert matrix == pytest.approx(hl.math.eye(9), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hl.TwoModeSum(0.3, wires=[0, 1])
        matrix = op.fock_matrix({0: 4, 1: 4})
        eye = hl.math.eye(16)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        lam = jnp.array(0.3)
        op = hl.TwoModeSum(lam, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 4, 1: 4})
        eye = hl.math.eye(16, like="jax")
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.SUM(x, wires=[0, 1])
            return op.fock_matrix({0: 4, 1: 4})

        x = jnp.array([0.123])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        # todo: don't have a good understanding of this gate to build a solid test
        def f(x):
            op = hl.SUM(x, wires=[0, 1])
            return op.fock_matrix({0: 4, 1: 4}).real  # ty:ignore[unresolved-attribute]

        x = jnp.array(0.123)
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))

    @pytest.mark.all_interfaces
    def test_heisenberg_rep(self, like):
        lam = hl.math.asarray(0.5, like=like)
        M = hl.TwoModeSum._heisenberg_rep([lam])
        assert hl.math.is_symplectic(M)
        assert hl.math.get_interface(M) == like
        assert hl.math.get_dtype_name(M) == "float64"

        # Check the expression in the mode basis, eq. B3
        mode_basis = hl.math.asarray(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, -lam / 2, lam / 2],
                [0, 0, 1, lam / 2, -lam / 2],
                [0, lam / 2, lam / 2, 1, 0],
                [0, lam / 2, lam / 2, 0, 1],
            ]
        )
        assert hl.math.to_fock_space(M) == pytest.approx(mode_basis)

        expected = hl.math.asarray(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, -lam],
                [0, lam, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        assert M == pytest.approx(expected, abs=1e-6)

    @pytest.mark.jax
    def test_heisenberg_rep_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(lam):
            return hl.TwoModeSum._heisenberg_rep([lam])

        f(jnp.array(0.5))  # errors if jit fails


@pytest.mark.unit
class TestBeamsplitter:
    def test_init(self):
        op = hl.Beamsplitter(0.5, 0.3, wires=[0, 1])
        assert op.name == "Beamsplitter"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.Beamsplitter(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.Beamsplitter)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_simplify(self):
        op = hl.BS(0, 0.123, wires=(0, 1))
        assert isinstance(op.simplify(), qp.Identity)

        op = hl.BS(0.123 + 4 * math.pi, 0.3 + 2 * math.pi, wires=[0, 1])
        simplified = op.simplify()
        assert simplified.parameters == pytest.approx([0.123, 0.3])

    def test_label(self):
        op = hl.Beamsplitter(0.5, 0.3, wires=[0, 1])
        assert op.label() == "BS"

    def test_fock_matrix_zero(self):
        op = hl.Beamsplitter(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 3, 1: 3})
        assert matrix == pytest.approx(hl.math.eye(9), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hl.Beamsplitter(math.pi / 4, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 4, 1: 4})
        eye = hl.math.eye(16)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    def test_fock_matrix_periodic(self):
        dim = 16
        n_cut = dim // 2
        wire_dims = {0: dim, 1: dim}
        op = hl.BS(0.123, 0.456, wires=(0, 1))
        mat = op.fock_matrix(wire_dims)
        op2 = hl.BS(0.123 + 4 * math.pi, 0.456 + 2 * math.pi, wires=(0, 1))
        mat2 = op2.fock_matrix(wire_dims)

        # Build index list for the inner subspace (n_total = n_a + n_b <= n_cut)
        inner = hl.math.asarray(
            [
                n_a * dim + n_b
                for n_a in range(dim)
                for n_b in range(dim)
                if n_a + n_b <= n_cut
            ]
        )
        assert mat[hl.math.ix_(inner, inner)] == pytest.approx(  # ty:ignore[not-subscriptable]
            mat2[hl.math.ix_(inner, inner)], abs=1e-10  # ty:ignore[not-subscriptable]
        )

    def test_fock_matrix_commutes(self):
        # Beamsplitter should commute with n_a + n_b
        dim = 8
        dims = {0: dim, 1: dim}
        op = hl.Beamsplitter(0.123, -0.456, wires=[0, 1])
        matrix = op.fock_matrix(dims)
        n_a = hl.N(0).fock_matrix(dims, wire_order=(0, 1))
        n_b = hl.N(1).fock_matrix(dims, wire_order=(0, 1))
        commutator = matrix @ (n_a + n_b) - (n_a + n_b) @ matrix  # ty:ignore[unsupported-operator]
        assert commutator == pytest.approx(0, abs=1e-6)

    def test_action_on_fock_operators(self):
        # Beamsplitter should transform a and b as follows (eq. 170, 171 of liu2026hybrid):
        # a' = cos(theta/2) * a - isin(theta/2)e^{i phi} * b
        # b' = cos(theta/2) * b - isin(theta/2)e^{-i phi} * a
        dim = 16
        n_cut = dim // 2
        dims = {0: dim, 1: dim}
        theta = 0.123
        phi = -0.456
        op = hl.Beamsplitter(theta, phi, wires=[0, 1])
        bs = op.fock_matrix(dims, wire_order=(0, 1))
        bsd = hl.math.dag(bs)
        a = hl.A(0).fock_matrix(dims, wire_order=(0, 1))
        b = hl.A(1).fock_matrix(dims, wire_order=(0, 1))
        a_prime = bsd @ a @ bs
        b_prime = bsd @ b @ bs
        expected_a_prime = (
            hl.math.cos(theta / 2) * a
            - 1j * hl.math.exp(1j * phi) * hl.math.sin(theta / 2) * b
        )
        expected_b_prime = (
            hl.math.cos(theta / 2) * b
            - 1j * hl.math.exp(-1j * phi) * hl.math.sin(theta / 2) * a
        )
        # Build index list for the inner subspace (n_total = n_a + n_b <= n_cut)
        inner = hl.math.asarray(
            [
                n_a * dim + n_b
                for n_a in range(dim)
                for n_b in range(dim)
                if n_a + n_b <= n_cut
            ]
        )
        assert a_prime[hl.math.ix_(inner, inner)] == pytest.approx(
            expected_a_prime[hl.math.ix_(inner, inner)], abs=1e-10
        )
        assert b_prime[hl.math.ix_(inner, inner)] == pytest.approx(
            expected_b_prime[hl.math.ix_(inner, inner)], abs=1e-10
        )

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(math.pi / 4)
        phi = jnp.array(0.0)
        op = hl.Beamsplitter(theta, phi, wires=[0, 1])  # ty:ignore[invalid-argument-type]
        matrix = op.fock_matrix({0: 4, 1: 4})
        eye = hl.math.eye(16, like="jax")
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.BS(*x, wires=(0, 1))
            return op.fock_matrix({0: 4, 1: 4})

        x = jnp.array([0.123, 0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 4, 1: 4}
        n2 = hl.N(1).fock_matrix(dims, wire_order=(0, 1))
        n2 = hl.math.asarray(n2, like="jax")
        state1 = jnp.array([0, 1, 0, 0])
        state2 = jnp.array([1, 0, 0, 0])
        state = hl.math.kron(state1, state2)  # |1, 0>

        # Function effectively can transfer excitation to qumode 2, and then we measure
        # that excitation level <n2>
        def f(x):
            op = hl.BS(*x, wires=(0, 1))
            mat = op.fock_matrix(dims)
            return hl.math.expectation_value(n2, mat @ state).real

        # x[0] increases the excitation in mode 2 up until pi, so the gradient should be
        # positive. As for x[1], it imparts relative phases and therefore should have grad 0
        x = jnp.array([0.123, 0.456])
        grad_fn = hl.math.grad(f)
        grad = grad_fn(x)
        assert grad[0] > 0
        assert grad[1] == pytest.approx(0)

        # pi should be local maximum
        x = jnp.array([jnp.pi, 0.456])
        grad = grad_fn(x)
        assert grad == pytest.approx(0)

    @pytest.mark.all_interfaces
    def test_heisenberg_rep(self, like):
        theta = hl.math.asarray(0.4, like=like)
        phi = hl.math.asarray(0.3, like=like)
        M = hl.Beamsplitter._heisenberg_rep([theta, phi])
        assert hl.math.is_symplectic(M)
        assert hl.math.get_interface(M) == like
        assert hl.math.get_dtype_name(M) == "float64"

        c, s = math.cos(0.4 / 2), math.sin(0.4 / 2)
        cp, sp = math.cos(0.3), math.sin(0.3)
        expected = hl.math.asarray(
            [
                [1, 0, 0, 0, 0],
                [0, c, 0, s * sp, s * cp],
                [0, 0, c, -s * cp, s * sp],
                [0, -s * sp, s * cp, c, 0],
                [0, -s * cp, -s * sp, 0, c],
            ]
        )
        assert M == pytest.approx(expected, abs=1e-6)

    @pytest.mark.jax
    def test_heisenberg_rep_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            return hl.Beamsplitter._heisenberg_rep([x[0], x[1]])

        f(jnp.array([0.4, 0.3]))  # errors if jit fails


@pytest.mark.unit
class TestTwoModeSqueezing:
    def test_init(self):
        op = hl.TwoModeSqueezing(0.5, 0.3, wires=[0, 1])
        assert op.name == "TwoModeSqueezing"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hl.TwoModeSqueezing(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.TwoModeSqueezing)
        assert adj_op.parameters[0] == 0.5
        assert adj_op.parameters[1] == pytest.approx(
            (0.3 + math.pi) % (2 * math.pi), abs=1e-6
        )

    def test_simplify(self):
        op = hl.TwoModeSqueezing(0, 0.3, wires=[0, 1])
        assert isinstance(op.simplify(), qp.Identity)

        op = hl.TwoModeSqueezing(0.123, 0.3 + 2 * math.pi, wires=[0, 1])
        assert op.simplify().parameters == pytest.approx([0.123, 0.3])

    def test_label(self):
        op = hl.TwoModeSqueezing(0.5, 0.3, wires=[0, 1])
        assert op.label() == "TMS"

    def test_fock_matrix_zero(self):
        op = hl.TwoModeSqueezing(0.0, 0.0, wires=[0, 1])
        matrix = op.fock_matrix({0: 3, 1: 3})
        assert matrix == pytest.approx(hl.math.eye(9), abs=1e-6)

    def test_fock_matrix_unitary(self):
        op = hl.TwoModeSqueezing(0.3, 0.5, wires=[0, 1])
        matrix = op.fock_matrix({0: 5, 1: 5})
        eye = hl.math.eye(25)
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        r = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hl.TwoModeSqueezing(r, phi, wires=[0, 1])
        matrix = op.fock_matrix({0: 5, 1: 5})
        eye = hl.math.eye(25, like="jax")
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.TMS(*x, wires=(0, 1))
            return op.fock_matrix({0: 4, 1: 4})

        x = jnp.array([0.123, 0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        # todo: don't have a good understanding of this gate to build a solid test
        def f(x):
            op = hl.TMS(*x, wires=(0, 1))
            return op.fock_matrix({0: 4, 1: 4}).real  # ty:ignore[unresolved-attribute]

        x = jnp.array([0.123, 0.456])
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))

    @pytest.mark.all_interfaces
    def test_heisenberg_rep(self, like):
        r = hl.math.asarray(0.3, like=like)
        phi = hl.math.asarray(0.4, like=like)
        M = hl.TwoModeSqueezing._heisenberg_rep([r, phi])
        assert hl.math.is_symplectic(M)
        assert hl.math.get_interface(M) == like
        assert hl.math.get_dtype_name(M) == "float64"

        ch, sh = math.cosh(0.3), math.sinh(0.3)
        cp, sp = math.cos(0.4), math.sin(0.4)
        expected = hl.math.asarray(
            [
                [1, 0, 0, 0, 0],
                [0, ch, 0, sh * cp, sh * sp],
                [0, 0, ch, sh * sp, -sh * cp],
                [0, sh * cp, sh * sp, ch, 0],
                [0, sh * sp, -sh * cp, 0, ch],
            ]
        )
        assert M == pytest.approx(expected, abs=1e-6)

    @pytest.mark.jax
    def test_heisenberg_rep_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            return hl.TwoModeSqueezing._heisenberg_rep(x)

        f(jnp.array([0.3, 0.4]))  # errors if jit fails
