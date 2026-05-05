# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qml
import pytest

import hybridlane as hqml
import hybridlane.sa as sa


@pytest.mark.unit
class TestQuadX:
    def test_init(self):
        op = hqml.QuadX(wires=0)
        assert op.name == "QuadX"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qml.wires.Wires(0)

    def test_diagonalizing_gates(self):
        op = hqml.QuadX(wires=0)
        assert op.diagonalizing_gates() == []

    def test_natural_basis(self):
        op = hqml.QuadX(wires=0)
        assert op.natural_basis == sa.ComputationalBasis.Position

    def test_position_spectrum(self):
        op = hqml.QuadX(wires=0)
        states = hqml.math.asarray([-1.0, 0.0, 1.0])
        result = op.position_spectrum(states)
        assert result == pytest.approx(states)

    def test_fock_matrix(self):
        op = hqml.QuadX(wires=0)
        matrix = op.fock_matrix({0: 3})
        lam = 1 / math.sqrt(2)
        expected = hqml.math.array(
            [
                [0, lam, 0],
                [lam, 0, math.sqrt(2) * lam],
                [0, math.sqrt(2) * lam, 0],
            ],
            dtype=complex,
        )
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestQuadP:
    def test_init(self):
        op = hqml.QuadP(wires=0)
        assert op.name == "QuadP"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qml.wires.Wires(0)

    def test_diagonalizing_gates(self):
        op = hqml.QuadP(wires=0)
        gates = op.diagonalizing_gates()
        assert len(gates) == 1
        assert isinstance(gates[0], hqml.Rotation)

    def test_natural_basis(self):
        op = hqml.QuadP(wires=0)
        assert op.natural_basis == sa.ComputationalBasis.Position

    def test_decomposition(self):
        op = hqml.QuadP(wires=0)
        decomp = op.decomposition()
        assert len(decomp) == 2
        assert isinstance(decomp[0], hqml.Rotation)
        assert decomp[0].parameters[0] == pytest.approx(-math.pi / 2)
        assert isinstance(decomp[1], hqml.QuadX)

    def test_fock_matrix(self):
        op = hqml.QuadP(wires=0)
        matrix = op.fock_matrix({0: 3})
        lam = 1 / math.sqrt(2)
        expected = hqml.math.array(
            [
                [0, 1j * lam, 0],
                [-1j * lam, 0, 1j * math.sqrt(2) * lam],
                [0, -1j * math.sqrt(2) * lam, 0],
            ]
        )
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestQuadOperator:
    def test_init(self):
        op = hqml.QuadOperator(0.5, wires=0)
        assert op.name == "QuadOperator"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires(0)

    def test_diagonalizing_gates(self):
        op = hqml.QuadOperator(0.5, wires=0)
        gates = op.diagonalizing_gates()
        assert len(gates) == 1
        assert isinstance(gates[0], hqml.Rotation)

    def test_natural_basis(self):
        op = hqml.QuadOperator(0.5, wires=0)
        assert op.natural_basis == sa.ComputationalBasis.Position

    def test_decomposition(self):
        phi = 0.5
        op = hqml.QuadOperator(phi, wires=0)
        decomp = op.decomposition()
        assert len(decomp) == 2
        assert isinstance(decomp[0], hqml.Rotation)
        assert decomp[0].parameters[0] == pytest.approx(phi)
        assert isinstance(decomp[1], hqml.QuadX)

        # Check the decomposition numerically
        dim = 16
        mat = op.fock_matrix({0: dim})
        r = decomp[0].fock_matrix({0: dim})
        x = hqml.X.compute_fock_matrix((dim,))
        actual = hqml.math.dag(r) @ mat @ r
        assert actual == pytest.approx(x)

    def test_fock_matrix_at_zero(self):
        op = hqml.QuadOperator(0.0, wires=0)
        matrix = op.fock_matrix({0: 4})
        x_matrix = hqml.QuadX(wires=0).fock_matrix({0: 4})
        assert matrix == pytest.approx(x_matrix)

    def test_fock_matrix_at_half_pi(self):
        op = hqml.QuadOperator(math.pi / 2, wires=0)
        matrix = op.fock_matrix({0: 4})
        p_matrix = hqml.QuadP(wires=0).fock_matrix({0: 4})
        assert matrix == pytest.approx(p_matrix)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        import jax.numpy as jnp

        phi = jnp.array(0.5)
        op = hqml.QuadOperator(phi, wires=0)
        matrix = op.fock_matrix({0: 4})
        assert hqml.math.get_interface(matrix) == "jax"
        matrixd = hqml.math.dag(matrix)
        assert matrix == pytest.approx(matrixd)


@pytest.mark.unit
class TestNumberOperator:
    def test_init(self):
        op = hqml.NumberOperator(wires=0)
        assert op.name == "NumberOperator"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qml.wires.Wires(0)

    def test_diagonalizing_gates(self):
        op = hqml.NumberOperator(wires=0)
        assert op.diagonalizing_gates() == []

    def test_natural_basis(self):
        op = hqml.NumberOperator(wires=0)
        assert op.natural_basis == sa.ComputationalBasis.Discrete

    def test_fock_spectrum(self):
        op = hqml.NumberOperator(wires=0)
        basis_states = hqml.math.arange(4)
        result = op.fock_spectrum(basis_states)
        assert hqml.math.all(result == basis_states)

    def test_fock_matrix(self):
        op = hqml.NumberOperator(wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.diag([0, 1, 2, 3])
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        op = hqml.NumberOperator(wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.diag([0, 1, 2, 3])
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestFockStateProjector:
    def test_init(self):
        op = hqml.FockStateProjector(hqml.math.asarray([1, 0]), wires=[0, 1])
        assert op.name == "FockStateProjector"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert hqml.math.array_equal(op.parameters[0], [1, 0])
        assert op.wires == qml.wires.Wires([0, 1])

    def test_num_wires(self):
        op1 = hqml.FockStateProjector(0, wires=[0])
        assert op1.num_wires == 1

        op3 = hqml.FockStateProjector([0, 1, 2], wires=[0, 1, 2])
        assert op3.num_wires == 3

    def test_diagonalizing_gates(self):
        op = hqml.FockStateProjector([1, 0], wires=[0, 1])
        assert op.diagonalizing_gates() == []

    def test_natural_basis(self):
        op = hqml.FockStateProjector([1, 0], wires=[0, 1])
        assert op.natural_basis == sa.ComputationalBasis.Discrete

    def test_fock_spectrum(self):
        op = hqml.FockStateProjector([1], wires=[0])
        basis_states = hqml.math.arange(4)
        result = op.fock_spectrum(basis_states)
        expected = hqml.math.asarray([0, 1, 0, 0])
        assert result == pytest.approx(expected)

        op = hqml.FockStateProjector([1, 2], wires=[0, 1])
        basis_states = hqml.math.asarray([[0, 1], [1, 2], [3, 2]])
        result = op.fock_spectrum(basis_states)
        expected = hqml.math.asarray([0, 1, 0])
        assert result == pytest.approx(expected)

    def test_fock_matrix(self):
        op = hqml.FockStateProjector([1], wires=[0])
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.diag([0, 1, 0, 0])
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    def test_fock_matrix_out_of_bounds(self):
        op = hqml.FockStateProjector(4, wires=[0])
        with pytest.raises(ValueError, match="cannot be constructed"):
            op.fock_matrix({0: 4})

    @pytest.mark.jax
    def test_fock_matrix_jax(self):
        op = hqml.FockStateProjector(hqml.math.asarray([2], like="jax"), wires=[0])
        matrix = op.fock_matrix({0: 4})
        assert hqml.math.get_interface(matrix) == "jax"
        expected = hqml.math.diag([0, 0, 1, 0])
        assert matrix == pytest.approx(expected)
