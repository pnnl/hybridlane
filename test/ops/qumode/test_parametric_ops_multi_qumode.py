# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
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
