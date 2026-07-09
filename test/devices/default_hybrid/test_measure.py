# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qp
import pytest
from pennylane.operation import Operator
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane.devices.default_hybrid.measure import (
    build_fock_matrix,
    diagonalize,
    is_diagonalizable,
)


@pytest.mark.parametrize(
    "obs,expected",
    [
        (hl.N(0), True),
        (hl.X(0), False),
        (hl.X(0) + 0.5 * hl.N(0), False),
        # (hl.N(0) + 0.123 * hl.N(0) ** 2, True),
        (qp.Z(0) @ hl.N(1), True),
        (hl.N(0) @ hl.X(0), False),
        (hl.N(0) @ hl.N(1) @ qp.X(2) @ qp.X(3), True),
        (hl.N(0) ** 3 @ qp.X(1), True),
    ],
)
@pytest.mark.unit
def test_is_diagonalizable(obs: Operator, expected: bool):
    assert is_diagonalizable(obs) == expected


@pytest.mark.unit
class TestDiagonalize:
    def test_spectral(self):
        obs = hl.FockStateProjector(1, wires=0)
        ev, gates = diagonalize(obs, (0,), {0: 4})

        assert ev == pytest.approx([0, 1, 0, 0])
        assert gates == []

        obs = hl.FockStateProjector([1, 2], wires=(0, 1))
        ev, gates = diagonalize(obs, (0, 1), {0: 3, 1: 3})

        assert ev == pytest.approx([0, 0, 0, 0, 0, 1, 0, 0, 0])
        assert gates == []

    def test_prod(self):
        obs = hl.N(0) @ hl.FockStateProjector(2, wires=1)
        ev, gates = diagonalize(obs, (0, 1), {0: 3, 1: 3})

        assert ev == pytest.approx([0, 0, 0, 0, 0, 1, 0, 0, 2])
        assert gates == []

        obs = qp.X(0) @ qp.X(1)
        ev, gates = diagonalize(obs, (0, 1), {0: 2, 1: 2})

        assert ev == pytest.approx([1, -1, -1, 1])
        assert set(gates) == {qp.H(0), qp.H(1)}  # ty:ignore[invalid-argument-type]

    def test_sprod(self):
        obs = 0.5 * hl.N(0)
        ev, gates = diagonalize(obs, (0,), {0: 4})

        assert ev == pytest.approx(hl.math.arange(4) / 2)
        assert gates == []

    def test_pow(self):
        obs = hl.N(0) ** 2
        ev, gates = diagonalize(obs, (0,), {0: 4})

        assert ev == pytest.approx(hl.math.arange(4) ** 2)
        assert gates == []


@pytest.mark.unit
@pytest.mark.all_interfaces
class TestBuildFockMatrix:
    def test_simple(self, like):
        wire_dims = {0: 4}
        obs = hl.QuadOperator(hl.math.array(0.5, like=like), wires=0)
        mat = build_fock_matrix(obs, Wires(0), wire_dims)
        assert hl.math.shape(mat) == (4, 4)

        expected_mat = obs.fock_matrix(wire_dims)
        assert mat == pytest.approx(expected_mat)

    def test_tensor_prod(self, like):
        wire_dims = {0: 3, 1: 4}
        obs = hl.X(0) @ hl.QuadOperator(hl.math.array(0.5, like=like), wires=1)
        mat = build_fock_matrix(obs, Wires((0, 1)), wire_dims)
        assert hl.math.shape(mat) == (12, 12)

        mat1 = hl.X(0).fock_matrix(wire_dims)
        mat2 = hl.QuadOperator(hl.math.array(0.5, like=like), wires=1).fock_matrix(
            wire_dims
        )
        expected_mat = hl.math.kron(mat1, mat2)
        assert mat == pytest.approx(expected_mat)

    def test_matrix_prod(self, like):
        wire_dims = {0: 5}
        obs = hl.QuadOperator(hl.math.array(0, like=like), wires=0) @ hl.X(0)
        mat = build_fock_matrix(obs, Wires(0), wire_dims)
        assert hl.math.shape(mat) == (5, 5)

        x_mat = hl.X(0).fock_matrix(wire_dims)
        expected_mat = hl.math.linalg.matrix_power(x_mat, 2)
        assert mat == pytest.approx(expected_mat)

    def test_pow(self, like):
        wire_dims = {0: 5}
        base_op = hl.N(0)
        obs = base_op**3
        mat = build_fock_matrix(obs, Wires(0), wire_dims)
        assert hl.math.shape(mat) == (5, 5)

        base_mat = base_op.fock_matrix(wire_dims)
        expected_mat = hl.math.linalg.matrix_power(base_mat, 3)
        assert mat == pytest.approx(expected_mat)

    def test_sprod(self, like):
        wire_dims = {0: 5}
        obs = 0.5 * hl.N(0)
        mat = build_fock_matrix(obs, Wires(0), wire_dims)
        assert hl.math.shape(mat) == (5, 5)

        base_mat = hl.N(0).fock_matrix(wire_dims)
        expected_mat = 0.5 * base_mat  # ty:ignore[unsupported-operator]
        assert mat == pytest.approx(expected_mat)

    def test_sum(self, like):
        wire_dims = {0: 5}
        obs = hl.N(0) + hl.N(0) ** 2
        mat = build_fock_matrix(obs, Wires(0), wire_dims)
        assert hl.math.shape(mat) == (5, 5)

        base_mat = hl.N(0).fock_matrix(wire_dims)
        expected_mat = base_mat + hl.math.linalg.matrix_power(base_mat, 2)
        assert mat == pytest.approx(expected_mat)

    def test_qubit(self, like):
        wire_dims = {0: 2}
        obs = qp.X(0)
        mat = build_fock_matrix(obs, Wires(0), wire_dims)
        assert hl.math.shape(mat) == (2, 2)

        expected_mat = obs.matrix()
        assert mat == pytest.approx(expected_mat)
