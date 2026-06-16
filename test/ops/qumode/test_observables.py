# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qp
import pytest

import hybridlane as hl
import hybridlane.wires as sa


@pytest.mark.unit
class TestQuadX:
    def test_init(self):
        op = hl.QuadX(wires=0)
        assert op.name == "QuadX"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qp.wires.Wires(0)

    def test_diagonalizing_gates(self):
        op = hl.QuadX(wires=0)
        assert op.diagonalizing_gates() == []

    def test_natural_basis(self):
        op = hl.QuadX(wires=0)
        assert op.natural_basis == sa.ComputationalBasis.Position

    def test_position_spectrum(self):
        op = hl.QuadX(wires=0)
        states = hl.math.asarray([-1.0, 0.0, 1.0])
        result = op.position_spectrum(states)
        assert result == pytest.approx(states)

    def test_fock_matrix(self):
        op = hl.QuadX(wires=0)
        matrix = op.fock_matrix({0: 3})
        lam = 1 / math.sqrt(2)
        expected = hl.math.array(
            [
                [0, lam, 0],
                [lam, 0, math.sqrt(2) * lam],
                [0, math.sqrt(2) * lam, 0],
            ],
            dtype=complex,
        )
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestQuadP:
    def test_init(self):
        op = hl.QuadP(wires=0)
        assert op.name == "QuadP"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qp.wires.Wires(0)

    def test_diagonalizing_gates(self):
        op = hl.QuadP(wires=0)
        gates = op.diagonalizing_gates()
        assert len(gates) == 1
        assert isinstance(gates[0], hl.Rotation)

    def test_natural_basis(self):
        op = hl.QuadP(wires=0)
        assert op.natural_basis == sa.ComputationalBasis.Position

    def test_decomposition(self):
        op = hl.QuadP(wires=0)
        decomp = op.decomposition()
        assert len(decomp) == 2
        assert isinstance(decomp[0], hl.Rotation)
        assert decomp[0].parameters[0] == pytest.approx(-math.pi / 2)
        assert isinstance(decomp[1], hl.QuadX)

    def test_fock_matrix(self):
        op = hl.QuadP(wires=0)
        matrix = op.fock_matrix({0: 3})
        lam = 1 / math.sqrt(2)
        expected = hl.math.array(
            [
                [0, 1j * lam, 0],
                [-1j * lam, 0, 1j * math.sqrt(2) * lam],
                [0, -1j * math.sqrt(2) * lam, 0],
            ]
        )
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestQuadOperator:
    def test_init(self):
        op = hl.QuadOperator(0.5, wires=0)
        assert op.name == "QuadOperator"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.wires == qp.wires.Wires(0)

    def test_diagonalizing_gates(self):
        op = hl.QuadOperator(0.5, wires=0)
        gates = op.diagonalizing_gates()
        assert len(gates) == 1
        assert isinstance(gates[0], hl.Rotation)

    def test_natural_basis(self):
        op = hl.QuadOperator(0.5, wires=0)
        assert op.natural_basis == sa.ComputationalBasis.Position

    def test_decomposition(self):
        phi = 0.5
        op = hl.QuadOperator(phi, wires=0)
        decomp = op.decomposition()
        assert len(decomp) == 2
        assert isinstance(decomp[0], hl.Rotation)
        assert decomp[0].parameters[0] == pytest.approx(phi)
        assert isinstance(decomp[1], hl.QuadX)

        # Check the decomposition numerically
        dim = 16
        mat = op.fock_matrix({0: dim})
        r = decomp[0].fock_matrix({0: dim})
        x = hl.X.compute_fock_matrix((dim,))
        actual = hl.math.dag(r) @ mat @ r
        assert actual == pytest.approx(x)

    def test_fock_matrix_at_zero(self):
        op = hl.QuadOperator(0.0, wires=0)
        matrix = op.fock_matrix({0: 4})
        x_matrix = hl.QuadX(wires=0).fock_matrix({0: 4})
        assert matrix == pytest.approx(x_matrix)

    def test_fock_matrix_at_half_pi(self):
        op = hl.QuadOperator(math.pi / 2, wires=0)
        matrix = op.fock_matrix({0: 4})
        p_matrix = hl.QuadP(wires=0).fock_matrix({0: 4})
        assert matrix == pytest.approx(p_matrix)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        phi = jnp.array(0.5)
        op = hl.QuadOperator(phi, wires=0)
        matrix = op.fock_matrix({0: 4})
        assert hl.math.get_interface(matrix) == "jax"
        matrixd = hl.math.dag(matrix)
        assert matrix == pytest.approx(matrixd)

    @pytest.mark.jax
    def test_fock_matrix_jit(self):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(phi):
            op = hl.QuadOperator(phi, wires=0)
            return op.fock_matrix({0: 4})

        phi = jnp.array(0.5)
        f(phi)  # errors if jit fails

    @pytest.mark.jax
    def test_fock_matrix_grad(self):
        import jax.numpy as jnp

        dims = {0: 4}
        coherent_state = hl.D(*jnp.array([0.123, 0]), wires=0).fock_matrix(dims)[:, 0]

        def f(x):
            mat = hl.QuadOperator(x, wires=0).fock_matrix(dims)
            return hl.math.expectation_value(mat, coherent_state).real

        x = jnp.array(0.123)
        grad_fn = hl.math.grad(f)
        grad = grad_fn(x)
        assert grad < 0

        x = jnp.array(0.0)
        grad = grad_fn(x)
        assert grad == pytest.approx(0)


@pytest.mark.unit
class TestNumberOperator:
    def test_init(self):
        op = hl.NumberOperator(wires=0)
        assert op.name == "NumberOperator"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qp.wires.Wires(0)

    def test_diagonalizing_gates(self):
        op = hl.NumberOperator(wires=0)
        assert op.diagonalizing_gates() == []

    def test_natural_basis(self):
        op = hl.NumberOperator(wires=0)
        assert op.natural_basis == sa.ComputationalBasis.Discrete

    def test_fock_spectrum(self):
        op = hl.NumberOperator(wires=0)
        basis_states = hl.math.arange(4)
        result = op.fock_spectrum(basis_states)
        assert hl.math.all(result == basis_states)

    def test_fock_matrix(self):
        op = hl.NumberOperator(wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hl.math.diag([0, 1, 2, 3])
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestFockStateProjector:
    def test_init(self):
        op = hl.FockStateProjector(hl.math.asarray([1, 0]), wires=[0, 1])
        assert op.name == "FockStateProjector"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert hl.math.array_equal(op.parameters[0], [1, 0])
        assert op.wires == qp.wires.Wires([0, 1])

    def test_num_wires(self):
        op1 = hl.FockStateProjector(0, wires=[0])
        assert op1.num_wires == 1

        op3 = hl.FockStateProjector([0, 1, 2], wires=[0, 1, 2])
        assert op3.num_wires == 3

    def test_diagonalizing_gates(self):
        op = hl.FockStateProjector([1, 0], wires=[0, 1])
        assert op.diagonalizing_gates() == []

    def test_natural_basis(self):
        op = hl.FockStateProjector([1, 0], wires=[0, 1])
        assert op.natural_basis == sa.ComputationalBasis.Discrete

    def test_fock_spectrum(self):
        op = hl.FockStateProjector([1], wires=[0])
        basis_states = hl.math.arange(4)
        result = op.fock_spectrum(basis_states)
        expected = hl.math.asarray([0, 1, 0, 0])
        assert result == pytest.approx(expected)

        op = hl.FockStateProjector([1, 2], wires=[0, 1])
        basis_states = hl.math.asarray(
            [
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
            ]
        )
        result = op.fock_spectrum(basis_states)
        expected = [0, 0, 0, 0, 0, 1, 0, 0, 0]
        assert result == pytest.approx(expected)

    def test_fock_matrix(self):
        op = hl.FockStateProjector(1, wires=[0])
        matrix = op.fock_matrix({0: 4})
        expected = hl.math.diag([0, 1, 0, 0])
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    def test_fock_matrix_out_of_bounds(self):
        op = hl.FockStateProjector(4, wires=[0])
        with pytest.raises(ValueError, match="cannot be constructed"):
            op.fock_matrix({0: 4})

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        op = hl.FockStateProjector(hl.math.asarray([2], like="jax"), wires=[0])
        matrix = op.fock_matrix({0: 4})
        assert hl.math.get_interface(matrix) == "jax"
        expected = hl.math.diag([0, 0, 1, 0])
        assert matrix == pytest.approx(expected)
