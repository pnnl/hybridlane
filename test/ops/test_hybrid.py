# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.pauli import PauliWord
from pennylane.bose import BoseWord
from hybridlane.ops.hybrid import (
    ConditionalRotation,
    ConditionalParity,
    SelectiveQubitRotation,
    SelectiveNumberArbitraryPhase,
    JaynesCummings,
    AntiJaynesCummings,
    Rabi,
    ConditionalDisplacement,
    ConditionalBeamsplitter,
    ConditionalTwoModeSqueezing,
    ConditionalTwoModeSum,
)


class TestConditionalRotation:
    def test_init(self):
        op = ConditionalRotation(0.5, wires=[0, 1])
        assert op.name == "ConditionalRotation"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = ConditionalRotation(0.5, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, ConditionalRotation)
        assert adj_op.parameters[0] == -0.5

    def test_pow(self):
        op = ConditionalRotation(0.5, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], ConditionalRotation)
        assert pow_op[0].parameters[0] == 1.0

    def test_simplify(self):
        op = ConditionalRotation(0, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

        op = ConditionalRotation(1e-9, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalParity:
    def test_init(self):
        op = ConditionalParity(wires=[0, 1])
        assert op.name == "ConditionalParity"
        assert op.num_params == 0
        assert op.num_wires == 2
        assert op.parameters == []
        assert op.wires == qml.wires.Wires([0, 1])

    def test_decomposition(self):
        op = ConditionalParity(wires=[0, 1])
        decomp = op.decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], ConditionalRotation)

    def test_adjoint(self):
        op = ConditionalParity(wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, ConditionalRotation)


class TestSelectiveQubitRotation:
    def test_init(self):
        op = SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        assert op.name == "SelectiveQubitRotation"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, SelectiveQubitRotation)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], SelectiveQubitRotation)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = SelectiveQubitRotation(0, 0.3, 1, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestSelectiveNumberArbitraryPhase:
    def test_init(self):
        op = SelectiveNumberArbitraryPhase(0.5, 1, wires=[0, 1])
        assert op.name == "SelectiveNumberArbitraryPhase"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = SelectiveNumberArbitraryPhase(0.5, 1, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, SelectiveNumberArbitraryPhase)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        op = SelectiveNumberArbitraryPhase(0.5, 1, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], SelectiveNumberArbitraryPhase)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        op = SelectiveNumberArbitraryPhase(0, 1, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestJaynesCummings:
    def test_init(self):
        op = JaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "JaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = JaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, JaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = JaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], JaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = JaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestAntiJaynesCummings:
    def test_init(self):
        op = AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "AntiJaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, AntiJaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], AntiJaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = AntiJaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestRabi:
    def test_init(self):
        op = Rabi(0.5, wires=[0, 1])
        assert op.name == "Rabi"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = Rabi(0.5, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, Rabi)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        op = Rabi(0.5, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], Rabi)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        op = Rabi(0, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalDisplacement:
    def test_init(self):
        op = ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op[0], ConditionalDisplacement)
        assert adj_op[0].parameters == [2.0, -0.3]

    def test_pow(self):
        op = ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], ConditionalDisplacement)
        assert pow_op[0].parameters == [0.25, 0.6]

    def test_simplify(self):
        op = ConditionalDisplacement(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalBeamsplitter:
    def test_init(self):
        op = ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalBeamsplitter"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, ConditionalBeamsplitter)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], ConditionalBeamsplitter)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = ConditionalBeamsplitter(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalTwoModeSqueezing:
    def test_init(self):
        op = ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSqueezing"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op[0], ConditionalTwoModeSqueezing)
        assert adj_op[0].parameters == [2.0, -0.3]

    def test_pow(self):
        op = ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], ConditionalTwoModeSqueezing)
        assert pow_op[0].parameters == [0.25, 0.6]

    def test_simplify(self):
        op = ConditionalTwoModeSqueezing(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalTwoModeSum:
    def test_init(self):
        op = ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSum"
        assert op.num_params == 1
        assert op.num_wires == 3
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, ConditionalTwoModeSum)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        op = ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], ConditionalTwoModeSum)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        op = ConditionalTwoModeSum(0, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)
