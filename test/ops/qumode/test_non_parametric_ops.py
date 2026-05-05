# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
class TestFourier:
    def test_init(self):
        op = hqml.Fourier(wires=0)
        assert op.name == "Fourier"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.parameters == []
        assert op.wires == qml.wires.Wires(0)

    def test_decomposition(self):
        op = hqml.Fourier(wires=0)
        decomp = op.decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], hqml.Rotation)

    def test_adjoint(self):
        op = hqml.Fourier(wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.Rotation)

    def test_label(self):
        op = hqml.Fourier(wires=0)
        assert op.label() == "F"

    def test_fock_matrix(self):
        op = hqml.Fourier(wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.diag([1, -1j, -1, 1j])
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestModeSwap:
    def test_init(self):
        op = hqml.ModeSwap(wires=[0, 1])
        assert op.name == "ModeSwap"
        assert op.num_params == 0
        assert op.num_wires == 2
        assert op.parameters == []
        assert op.wires == qml.wires.Wires([0, 1])

    def test_decomposition(self):
        op = hqml.ModeSwap(wires=[0, 1])
        decomp = op.decomposition()
        assert len(decomp) == 3
        assert isinstance(decomp[0], hqml.Beamsplitter)
        assert isinstance(decomp[1], hqml.Rotation)
        assert isinstance(decomp[2], hqml.Rotation)

        # Occupy only half the modes to avoid truncation problems
        dim = 16
        state1 = hqml.math.random.randn(dim)
        state1[-dim // 2 :] = 0
        state2 = hqml.math.random.randn(dim)
        state2[-dim // 2 :] = 0
        state = hqml.math.kron(state1, state2)

        dims = {0: dim, 1: dim}
        swap = op.fock_matrix(dims, wire_order=(0, 1))
        expected_state = swap @ state

        actual_state = state
        for op in decomp:
            actual_state = op.fock_matrix(dims, wire_order=(0, 1)) @ actual_state

        assert actual_state == pytest.approx(expected_state)

    def test_adjoint(self):
        op = hqml.ModeSwap(wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ModeSwap)

    def test_pow(self):
        op = hqml.ModeSwap(wires=[0, 1])
        for i in range(1, 6):
            if i % 2 == 0:
                assert isinstance(op.pow(i)[0], qml.Identity)
            else:
                assert isinstance(op.pow(i)[0], hqml.ModeSwap)

    def test_fock_matrix(self):
        op = hqml.ModeSwap(wires=[0, 1])
        matrix = op.fock_matrix({0: 2, 1: 2})
        expected = qml.SWAP.compute_matrix()
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestCreationOp:
    def test_init(self):
        op = hqml.CreationOp(wires=0)
        assert op.name == "CreationOp"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qml.wires.Wires(0)

    def test_adjoint(self):
        op = hqml.CreationOp(wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.AnnihilationOp)
        assert adj_op.wires == op.wires

    def test_fock_matrix(self):
        op = hqml.CreationOp(wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [0, math.sqrt(2), 0, 0],
                [0, 0, math.sqrt(3), 0],
            ],
            dtype=complex,
        )
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)


@pytest.mark.unit
class TestAnnihilationOp:
    def test_init(self):
        op = hqml.AnnihilationOp(wires=0)
        assert op.name == "AnnihilationOp"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qml.wires.Wires(0)

    def test_adjoint(self):
        op = hqml.AnnihilationOp(wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.CreationOp)
        assert adj_op.wires == op.wires

    def test_fock_matrix(self):
        op = hqml.AnnihilationOp(wires=0)
        matrix = op.fock_matrix({0: 4})
        expected = hqml.math.array(
            [
                [0, 1, 0, 0],
                [0, 0, math.sqrt(2), 0],
                [0, 0, 0, math.sqrt(3)],
                [0, 0, 0, 0],
            ],
            dtype=complex,
        )
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)
