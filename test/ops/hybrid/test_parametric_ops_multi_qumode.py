# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
class TestConditionalBeamsplitter:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalBeamsplitter"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalBeamsplitter)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalBeamsplitter)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.ConditionalBeamsplitter(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


@pytest.mark.unit
class TestConditionalTwoModeSqueezing:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSqueezing"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op[0], hqml.ConditionalTwoModeSqueezing)
        assert adj_op[0].parameters == [-0.5, 0.3]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalTwoModeSqueezing)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.ConditionalTwoModeSqueezing(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


@pytest.mark.unit
class TestConditionalTwoModeSum:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSum"
        assert op.num_params == 1
        assert op.num_wires == 3
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalTwoModeSum)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalTwoModeSum)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.ConditionalTwoModeSum(0, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)
