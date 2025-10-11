# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import pennylane as qml
from pennylane import numpy as np

import hybridlane as hqml


class TestTwoModeSum:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.TwoModeSum(0.5, wires=[0, 1])
        assert op.name == "TwoModeSum"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.TwoModeSum(0.5, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.TwoModeSum)
        assert adj_op.parameters[0] == -0.5

    def test_pow(self):
        """Test the pow method."""
        op = hqml.TwoModeSum(0.5, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.TwoModeSum)
        assert pow_op[0].parameters[0] == 1.0

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.TwoModeSum(0, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

        op = hqml.TwoModeSum(1e-9, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestModeSwap:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.ModeSwap(wires=[0, 1])
        assert op.name == "ModeSwap"
        assert op.num_params == 0
        assert op.num_wires == 2
        assert op.parameters == []
        assert op.wires == qml.wires.Wires([0, 1])

    def test_decomposition(self):
        """Test the decomposition method."""
        op = hqml.ModeSwap(wires=[0, 1])
        decomp = op.decomposition()
        assert len(decomp) == 3
        assert isinstance(decomp[0], hqml.Beamsplitter)
        assert isinstance(decomp[1], hqml.Rotation)
        assert isinstance(decomp[2], hqml.Rotation)

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.ModeSwap(wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ModeSwap)

    def test_pow(self):
        """Test the pow method."""
        op = hqml.ModeSwap(wires=[0, 1])
        pow_op_even = op.pow(2)
        assert isinstance(pow_op_even[0], qml.Identity)

        pow_op_odd = op.pow(3)
        assert isinstance(pow_op_odd[0], hqml.ModeSwap)


class TestFourier:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.Fourier(wires=0)
        assert op.name == "Fourier"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.parameters == []
        assert op.wires == qml.wires.Wires(0)

    def test_decomposition(self):
        """Test the decomposition method."""
        op = hqml.Fourier(wires=0)
        decomp = op.decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], hqml.Rotation)

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.Fourier(wires=0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.Rotation)


class TestQuadX:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.QuadX(wires=0)
        assert op.name == "QuadX"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qml.wires.Wires(0)

    def test_diagonalizing_gates(self):
        """Test the diagonalizing_gates method."""
        op = hqml.QuadX(wires=0)
        assert op.diagonalizing_gates() == []


class TestQuadP:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.QuadP(wires=0)
        assert op.name == "QuadP"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qml.wires.Wires(0)

    def test_diagonalizing_gates(self):
        """Test the diagonalizing_gates method."""
        op = hqml.QuadP(wires=0)
        gates = op.diagonalizing_gates()
        assert len(gates) == 1
        assert isinstance(gates[0], hqml.Rotation)


class TestQuadOperator:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.QuadOperator(0.5, wires=0)
        assert op.name == "QuadOperator"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires(0)

    def test_diagonalizing_gates(self):
        """Test the diagonalizing_gates method."""
        op = hqml.QuadOperator(0.5, wires=0)
        gates = op.diagonalizing_gates()
        assert len(gates) == 1
        assert isinstance(gates[0], hqml.Rotation)


class TestNumberOperator:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.NumberOperator(wires=0)
        assert op.name == "NumberOperator"
        assert op.num_params == 0
        assert op.num_wires == 1
        assert op.wires == qml.wires.Wires(0)

    def test_diagonalizing_gates(self):
        """Test the diagonalizing_gates method."""
        op = hqml.NumberOperator(wires=0)
        assert op.diagonalizing_gates() == []


class TestFockStateProjector:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.FockStateProjector(np.array([1, 0]), wires=[0, 1])
        assert op.name == "FockStateProjector"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert np.array_equal(op.parameters[0], np.array([1, 0]))
        assert op.wires == qml.wires.Wires([0, 1])

    def test_diagonalizing_gates(self):
        """Test the diagonalizing_gates method."""
        op = hqml.FockStateProjector(np.array([1, 0]), wires=[0, 1])
        assert op.diagonalizing_gates() == []
