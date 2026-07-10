# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: N806
import math

import pennylane as qp
import pytest

import hybridlane as hl


@pytest.mark.unit
class TestFourier:
    def test_init(self):
        op = hl.Fourier(wires=0)
        assert op.name == "Fourier"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.parameters == []
        assert op.wires == qp.wires.Wires(0)

    def test_decomposition(self):
        op = hl.Fourier(wires=0)
        decomp = op.decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], hl.Rotation)

    def test_adjoint(self):
        op = hl.Fourier(wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.Rotation)

    def test_label(self):
        op = hl.Fourier(wires=0)
        assert op.label() == "F"

    def test_fock_matrix(self):
        op = hl.Fourier(wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hl.math.diag([1, -1j, -1, 1j])
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    def test_heisenberg_rep(self):
        M = hl.Fourier._heisenberg_rep([])
        assert hl.math.is_symplectic(M)
        assert hl.math.get_interface(M) == "numpy"
        assert hl.math.get_dtype_name(M) == "float64"

        # see box IV.2 of liu2026hybrid for these relations. the fourier
        # gate should perform a clockwise rotation by pi/2
        expected = hl.math.asarray(
            [
                [1, 0, 0],
                [0, 0, 1],  # p -> x
                [0, -1, 0],  # x -> -p
            ]
        )
        assert pytest.approx(expected) == M


@pytest.mark.unit
class TestModeSwap:
    def test_init(self):
        op = hl.ModeSwap(wires=[0, 1])
        assert op.name == "ModeSwap"
        assert op.num_params == 0
        assert op.num_wires == 2
        assert op.parameters == []
        assert op.wires == qp.wires.Wires([0, 1])

    def test_decomposition(self):
        op = hl.ModeSwap(wires=[0, 1])
        decomp = op.decomposition()
        assert len(decomp) == 3
        assert isinstance(decomp[0], hl.Beamsplitter)
        assert isinstance(decomp[1], hl.Rotation)
        assert isinstance(decomp[2], hl.Rotation)

        # Occupy only half the modes to avoid truncation problems
        dim = 16
        state1 = hl.math.random.randn(dim)
        state1[-dim // 2 :] = 0
        state2 = hl.math.random.randn(dim)
        state2[-dim // 2 :] = 0
        state = hl.math.kron(state1, state2)

        dims = {0: dim, 1: dim}
        swap = op.fock_matrix(dims, wire_order=(0, 1))
        expected_state = swap @ state

        actual_state = state
        for op in decomp:
            actual_state = op.fock_matrix(dims, wire_order=(0, 1)) @ actual_state  # ty:ignore[unresolved-attribute]

        assert actual_state == pytest.approx(expected_state)

    def test_adjoint(self):
        op = hl.ModeSwap(wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.ModeSwap)

    def test_pow(self):
        op = hl.ModeSwap(wires=[0, 1])
        for i in range(1, 6):
            if i % 2 == 0:
                assert isinstance(op.pow(i)[0], qp.Identity)
            else:
                assert isinstance(op.pow(i)[0], hl.ModeSwap)

    def test_fock_matrix(self):
        op = hl.ModeSwap(wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 2})
        expected = qp.SWAP.compute_matrix()
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)

    def test_heisenberg_rep(self):
        M = hl.ModeSwap._heisenberg_rep([])
        assert hl.math.is_symplectic(M)
        assert hl.math.get_interface(M) == "numpy"
        assert hl.math.get_dtype_name(M) == "float64"

        expected = hl.math.asarray(
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )
        assert pytest.approx(expected, abs=1e-6) == M


@pytest.mark.unit
class TestCreationOp:
    def test_init(self):
        op = hl.CreationOp(wires=0)
        assert op.name == "CreationOp"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qp.wires.Wires(0)

    def test_adjoint(self):
        op = hl.CreationOp(wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.AnnihilationOp)
        assert adj_op.wires == op.wires

    def test_fock_matrix(self):
        op = hl.CreationOp(wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hl.math.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, math.sqrt(2), 0, 0],
                [0, 0, math.sqrt(3), 0],
            ],
            dtype=complex,
        )
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestAnnihilationOp:
    def test_init(self):
        op = hl.AnnihilationOp(wires=0)
        assert op.name == "AnnihilationOp"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qp.wires.Wires(0)

    def test_adjoint(self):
        op = hl.AnnihilationOp(wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.CreationOp)
        assert adj_op.wires == op.wires

    def test_fock_matrix(self):
        op = hl.AnnihilationOp(wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hl.math.array(
            [
                [0, 1, 0, 0],
                [0, 0, math.sqrt(2), 0],
                [0, 0, 0, math.sqrt(3)],
                [0, 0, 0, 0],
            ],
            dtype=complex,
        )
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)
