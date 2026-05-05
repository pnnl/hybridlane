# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qml
import pytest
from pennylane import numpy as np

import hybridlane as hqml


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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
