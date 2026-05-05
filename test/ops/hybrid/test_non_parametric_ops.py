# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
class TestConditionalParity:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.ConditionalParity(wires=[0, 1])
        assert op.name == "ConditionalParity"
        assert op.num_params == 0
        assert op.num_wires == 2
        assert op.parameters == []
        assert op.wires == qml.wires.Wires([0, 1])

    def test_decomposition(self):
        """Test the decomposition method."""
        op = hqml.ConditionalParity(wires=[0, 1])
        decomp = op.decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], hqml.ConditionalRotation)

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.ConditionalParity(wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.ConditionalRotation)
