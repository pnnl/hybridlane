# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
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


@pytest.mark.unit
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
