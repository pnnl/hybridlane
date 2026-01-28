# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import pennylane as qml

import hybridlane as hqml


class TestConditionalRotation:
    def test_init(self):
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        assert op.name == "ConditionalRotation"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalRotation)
        assert adj_op.parameters[0] == -0.5

    def test_pow(self):
        op = hqml.ConditionalRotation(0.5, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalRotation)
        assert pow_op[0].parameters[0] == 1.0

    def test_simplify(self):
        op = hqml.ConditionalRotation(0, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

        op = hqml.ConditionalRotation(1e-9, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalParity:
    def test_init(self):
        op = hqml.ConditionalParity(wires=[0, 1])
        assert op.name == "ConditionalParity"
        assert op.num_params == 0
        assert op.num_wires == 2
        assert op.parameters == []
        assert op.wires == qml.wires.Wires([0, 1])

    def test_decomposition(self):
        op = hqml.ConditionalParity(wires=[0, 1])
        decomp = op.decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], hqml.ConditionalRotation)

    def test_adjoint(self):
        op = hqml.ConditionalParity(wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalRotation)


class TestSelectiveQubitRotation:
    def test_init(self):
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        assert op.name == "SelectiveQubitRotation"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.SelectiveQubitRotation)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.SelectiveQubitRotation(0.5, 0.3, 1, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.SelectiveQubitRotation)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.SelectiveQubitRotation(0, 0.3, 1, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestSelectiveNumberArbitraryPhase:
    def test_init(self):
        op = hqml.SelectiveNumberArbitraryPhase(0.5, 1, wires=[0, 1])
        assert op.name == "SelectiveNumberArbitraryPhase"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.SelectiveNumberArbitraryPhase(0.5, 1, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.SelectiveNumberArbitraryPhase)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        op = hqml.SelectiveNumberArbitraryPhase(0.5, 1, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.SelectiveNumberArbitraryPhase)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        op = hqml.SelectiveNumberArbitraryPhase(0, 1, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestJaynesCummings:
    def test_init(self):
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "JaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.JaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.JaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.JaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.JaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestAntiJaynesCummings:
    def test_init(self):
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        assert op.name == "AntiJaynesCummings"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.AntiJaynesCummings)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.AntiJaynesCummings(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.AntiJaynesCummings)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.AntiJaynesCummings(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestRabi:
    def test_init(self):
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        assert op.name == "Rabi"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.Rabi)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.Rabi(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.Rabi)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.Rabi(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalDisplacement:
    def test_init(self):
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        assert op.name == "ConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalDisplacement)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.ConditionalDisplacement(0.5, 0.3, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalDisplacement)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.ConditionalDisplacement(0, 0.3, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalBeamsplitter:
    def test_init(self):
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalBeamsplitter"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalBeamsplitter)
        assert adj_op.parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.ConditionalBeamsplitter(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalBeamsplitter)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.ConditionalBeamsplitter(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalTwoModeSqueezing:
    def test_init(self):
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSqueezing"
        assert op.num_params == 2
        assert op.num_wires == 3
        assert op.parameters == [0.5, 0.3]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op[0], hqml.ConditionalTwoModeSqueezing)
        assert adj_op[0].parameters == [-0.5, 0.3]

    def test_pow(self):
        op = hqml.ConditionalTwoModeSqueezing(0.5, 0.3, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalTwoModeSqueezing)
        assert pow_op[0].parameters == [1.0, 0.3]

    def test_simplify(self):
        op = hqml.ConditionalTwoModeSqueezing(0, 0.3, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestConditionalTwoModeSum:
    def test_init(self):
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        assert op.name == "ConditionalTwoModeSum"
        assert op.num_params == 1
        assert op.num_wires == 3
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1, 2])

    def test_adjoint(self):
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalTwoModeSum)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        op = hqml.ConditionalTwoModeSum(0.5, wires=[0, 1, 2])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.ConditionalTwoModeSum)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        op = hqml.ConditionalTwoModeSum(0, wires=[0, 1, 2])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestEchoedConditionalDisplacement:
    def test_init(self):
        op = hqml.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        assert op.name == "EchoedConditionalDisplacement"
        assert op.num_params == 2
        assert op.num_wires == 2
        assert op.parameters == [0.5, 0]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.EchoedConditionalDisplacement(0.5, 0, wires=[0, 1])
        adj_op = op.adjoint()[0]
        assert adj_op.parameters == [-0.5, 0]

    def test_pow(self):
        op = hqml.EchoedConditionalDisplacement(0.5, 0.123, wires=[0, 1])
        pow_op = op.pow(4)[0]
        assert pow_op.parameters == [2, 0.123]

    def test_simplify(self):
        op = hqml.EchoedConditionalDisplacement(0, 0.123, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)
