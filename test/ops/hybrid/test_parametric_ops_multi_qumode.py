# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
class TestConditionalBeamsplitter:
    def test_init(self):
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalBeamsplitter"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalBeamsplitter)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalBeamsplitter)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.ConditionalBeamsplitter(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        assert op.label() == "CBS"

    def test_fock_matrix_zero(self):
        op = hqml.ConditionalBeamsplitter(0.0, 0.0, wires=[0, 1, 2])
        matrix = op.fock_matrix({0: 2, 1: 3, 2: 3})
        assert matrix == pytest.approx(hqml.math.eye(18), abs=1e-6)

    def test_fock_matrix(self):
        dims = {0: 2, 1: 4, 2: 4}
        bs = hqml.BS(0.5, 0.3, wires=(1, 2)).fock_matrix(dims)
        op = hqml.CBS(0.5, 0.3, wires=[0, 1, 2])
        matrix = op.fock_matrix(dims)
        assert matrix == pytest.approx(hqml.math.block_diag([bs, hqml.math.dag(bs)]))

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        theta = jnp.array(0.5)
        phi = jnp.array(0.3)
        op = hqml.ConditionalBeamsplitter(theta, phi, wires=[0, 1, 2])
        matrix = op.fock_matrix({0: 2, 1: 4, 2: 4})
        eye = hqml.math.eye(32, like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.CBS(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hqml.CBS(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4}).real

        x = jnp.array([0.123, -0.456])
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalTwoModeSqueezing:
    def test_init(self):
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSqueezing"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op[0], hqml.ConditionalTwoModeSqueezing)
        assert adj_op[0].parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalTwoModeSqueezing)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.ConditionalTwoModeSqueezing(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        assert op.label() == "CTMS"

    def test_fock_matrix(self):
        dims = {0: 2, 1: 4, 2: 4}
        tms = hqml.TMS(0.3, 0.5, wires=(1, 2)).fock_matrix(dims)
        op = hqml.CTMS(0.3, 0.5, wires=[0, 1, 2])
        matrix = op.fock_matrix(dims)
        assert matrix == pytest.approx(hqml.math.block_diag([tms, hqml.math.dag(tms)]))

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        r = jnp.array(0.3)
        phi = jnp.array(0.5)
        op = hqml.ConditionalTwoModeSqueezing(r, phi, wires=[0, 1, 2])
        matrix = op.fock_matrix({0: 2, 1: 4, 2: 4})
        eye = hqml.math.eye(32, like="jax")
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.CTMS(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4})

        x = jnp.array([0.123, -0.456])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hqml.CTMS(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4}).real

        x = jnp.array([0.123, -0.456])
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))


@pytest.mark.unit
class TestConditionalTwoModeSum:
    def test_init(self):
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSum"
        assert op.num_params == 1
        assert op.num_wires == 3
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalTwoModeSum)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalTwoModeSum)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        op = hqml.ConditionalTwoModeSum(0, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_label(self):
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        assert op.label() == "CSUM"

    def test_fock_matrix(self):
        dims = {0: 2, 1: 4, 2: 4}
        sum = hqml.SUM(0.3, wires=(1, 2)).fock_matrix(dims)
        op = hqml.CSUM(0.3, wires=[0, 1, 2])
        matrix = op.fock_matrix(dims)
        assert matrix == pytest.approx(hqml.math.block_diag([sum, hqml.math.dag(sum)]))

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        lam = jnp.array(0.3)
        op = hqml.ConditionalTwoModeSum(lam, wires=[0, 1, 2])
        matrix = op.fock_matrix({0: 2, 1: 4, 2: 4})
        eye = hqml.math.eye(32, like="jax")
        assert matrix @ hqml.math.dag(matrix) == pytest.approx(eye, abs=1e-6)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(x):
            op = hqml.CSUM(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4})

        x = jnp.array([0.123])
        f(x)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        def f(x):
            op = hqml.CSUM(*x, wires=(0, 1, 2))
            return op.fock_matrix({0: 2, 1: 4, 2: 4}).real

        x = jnp.array([0.123])
        grad_fn = hqml.math.jacobian(f)
        grad = grad_fn(x)
        assert not jnp.any(jnp.isnan(grad))
