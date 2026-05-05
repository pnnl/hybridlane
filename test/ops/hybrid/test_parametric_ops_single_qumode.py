# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
class TestConditionalRotation:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        assert op.name == "ConditionalRotation"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalRotation)
        assert adj_op.parameters[0] == -0.5

    def test_pow(self):
        """Test the pow method."""
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalRotation)
        assert pow_op[0].parameters[0] == 1.0

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.ConditionalRotation(0, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

        op = hqml.ConditionalRotation(1e-9, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


@pytest.mark.unit
class TestSelectiveQubitRotation:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        assert op.name == "SelectiveQubitRotation"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.SelectiveQubitRotation)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.SelectiveQubitRotation)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.SelectiveQubitRotation(0, 0.3, 1, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


@pytest.mark.unit
class TestJaynesCummings:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "JaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.JaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.JaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.JaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


@pytest.mark.unit
class TestAntiJaynesCummings:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "AntiJaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.AntiJaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.AntiJaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.AntiJaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


@pytest.mark.unit
class TestRabi:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        assert op.name == "Rabi"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.Rabi)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.Rabi)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.Rabi(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


@pytest.mark.unit
class TestConditionalDisplacement:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalDisplacement)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalDisplacement)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.ConditionalDisplacement(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


@pytest.mark.unit
class TestEchoedConditionalDisplacement:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        assert op.name == "EchoedConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        adj_op = op.adjoint()[0]
        assert adj_op.parameters == [-0.5, 0]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.EchoedConditionalDisplacement(0.5, 0.123, wires=[0, 1])
        pow_op = op.pow(4)[0]
        assert pow_op.parameters == [2, 0.123]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.EchoedConditionalDisplacement(0, 0.123, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)
