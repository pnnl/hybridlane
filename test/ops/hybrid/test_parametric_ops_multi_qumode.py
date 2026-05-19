# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

import pennylane as qp
import pytest

import hybridlane as hl


@pytest.mark.unit
class TestConditionalBeamsplitter:
    def test_init(self):
        op = hl.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalBeamsplitter"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = hl.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.ConditionalBeamsplitter)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.ConditionalBeamsplitter)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.ConditionalBeamsplitter(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        assert op.label() == "CBS"

    def test_fock_matrix_zero(self):
        op = hl.ConditionalBeamsplitter(0.0, 0.0, wires=[0, 1, 2])
        matrix = op.fock_matrix({0: 2, 1: 3, 2: 3})
        assert matrix == pytest.approx(hl.math.eye(18), abs=1e-6)

    def test_fock_matrix(self):
        dims = {0: 2, 1: 4, 2: 4}
        bs = hl.BS(0.5, 0.3, wires=(1, 2)).fock_matrix(dims)
        op = hl.CBS(0.5, 0.3, wires=[0, 1, 2])
        matrix = op.fock_matrix(dims)
        assert matrix == pytest.approx(hl.math.block_diag([bs, hl.math.dag(bs)]))

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hl.ConditionalBeamsplitter(theta, phi, wires=[0, 1, 2])
        matrix = op.fock_matrix({0: 2, 1: 4, 2: 4})
        eye = hl.math.eye(32, like="jax")
        assert hl.math.get_interface(matrix) == "jax"
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.CBS(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hl.CBS(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4}).real

        x = jnp.array([0.123, -0.456])
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalTwoModeSqueezing:
    def test_init(self):
        op = hl.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSqueezing"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qp.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = hl.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op[0], hl.ConditionalTwoModeSqueezing)
        assert adj_op[0].parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hl.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.ConditionalTwoModeSqueezing)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hl.ConditionalTwoModeSqueezing(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        assert op.label() == "CTMS"

    def test_fock_matrix(self):
        dims = {0: 2, 1: 4, 2: 4}
        tms = hl.TMS(0.3, 0.5, wires=(1, 2)).fock_matrix(dims)
        op = hl.CTMS(0.3, 0.5, wires=[0, 1, 2])
        matrix = op.fock_matrix(dims)
        assert matrix == pytest.approx(hl.math.block_diag([tms, hl.math.dag(tms)]))

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        r = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hl.ConditionalTwoModeSqueezing(r, phi, wires=[0, 1, 2])
        matrix = op.fock_matrix({0: 2, 1: 4, 2: 4})
        eye = hl.math.eye(32, like="jax")
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.CTMS(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hl.CTMS(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4}).real

        x = jnp.array([0.123, -0.456])
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalTwoModeSum:
    def test_init(self):
        op = hl.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSum"
        assert op.num_params == 1
        assert op.num_wires == 3
        assert op.parameters == [0.5]
        assert op.wires == qp.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = hl.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.ConditionalTwoModeSum)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        op = hl.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hl.ConditionalTwoModeSum)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        op = hl.ConditionalTwoModeSum(0, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qp.Identity)

    def test_label(self):
        op = hl.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        assert op.label() == "CSUM"

    def test_fock_matrix(self):
        dims = {0: 2, 1: 4, 2: 4}
        sum = hl.SUM(0.3, wires=(1, 2)).fock_matrix(dims)
        op = hl.CSUM(0.3, wires=[0, 1, 2])
        matrix = op.fock_matrix(dims)
        assert matrix == pytest.approx(hl.math.block_diag([sum, hl.math.dag(sum)]))

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        lam = jnp.array(0.3)
        op = hl.ConditionalTwoModeSum(lam, wires=[0, 1, 2])
        matrix = op.fock_matrix({0: 2, 1: 4, 2: 4})
        eye = hl.math.eye(32, like="jax")
        assert matrix @ hl.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hl.CSUM(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4})

        x = jnp.array([0.123])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hl.CSUM(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4}).real

        x = jnp.array([0.123])
        grad_fn = hl.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))
