# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import numpy as np
import pennylane as qml

import hybridlane as hqml


class TestBeamsplitter:
    def test_init(self):
        op = hqml.Beamsplitter(0.5, 0.123, [0, 1])
        assert op.name == "Beamsplitter"
        assert op.parameters == [0.5, 0.123]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.Beamsplitter(0.5, 0.123, [0, 1])
        adj_op = op.adjoint()
        assert adj_op.parameters == [-0.5, 0.123]

    def test_pow(self):
        op = hqml.Beamsplitter(0.5, 0.123, [0, 1])
        pow_op = op.pow(4)
        assert len(pow_op) == 1
        assert pow_op[0].parameters == [2, 0.123]

    def test_heisenberg_rep(self):
        op = hqml.Beamsplitter(np.pi, 0, [0, 1])
        S = op.heisenberg_tr(op.wires)
        assert hqml.heisenberg.is_symplectic(S)
        Sf = hqml.heisenberg.to_fock_space(S)
        assert hqml.heisenberg.is_symplectic(Sf)
        assert np.allclose(np.diag(Sf), [1, 0, 0, 0, 0])


class TestTwoModeSqueezing:
    def test_init(self):
        op = hqml.TwoModeSqueezing(0.5, 0.123, [0, 1])
        assert op.name == "TwoModeSqueezing"
        assert op.wires == qml.wires.Wires([0, 1])
        assert op.parameters == [0.5, 0.123]

    def test_adjoint(self):
        op = hqml.TwoModeSqueezing(0.5, 0.123, [0, 1])
        adj_op = op.adjoint()
        assert adj_op.parameters == [-0.5, 0.123]

    def test_pow(self):
        op = hqml.TwoModeSqueezing(0.5, 0.123, [0, 1])
        pow_op = op.pow(4)
        assert len(pow_op) == 1
        assert pow_op[0].parameters == [2, 0.123]

    def test_heisenberg_rep(self):
        op = hqml.TwoModeSqueezing(0.5, 0.123, [0, 1])
        S = op.heisenberg_tr(op.wires)
        assert hqml.heisenberg.is_symplectic(S)


class TestTwoModeSum:
    def test_init(self):
        op = hqml.TwoModeSum(0.5, wires=[0, 1])
        assert op.name == "TwoModeSum"
        assert op.num_params == 1
        assert op.num_wires == 2
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires([0, 1])

    def test_adjoint(self):
        op = hqml.TwoModeSum(0.5, wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.TwoModeSum)
        assert adj_op.parameters[0] == -0.5

    def test_pow(self):
        op = hqml.TwoModeSum(0.5, wires=[0, 1])
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.TwoModeSum)
        assert pow_op[0].parameters[0] == 1.0

    def test_simplify(self):
        op = hqml.TwoModeSum(0, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

        op = hqml.TwoModeSum(1e-9, wires=[0, 1])
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_heisenberg_rep(self):
        op = hqml.TwoModeSum(0.5, [0, 1])
        S = op.heisenberg_tr(op.wires)
        assert hqml.heisenberg.is_symplectic(S)
