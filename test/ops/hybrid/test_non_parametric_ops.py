# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
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

    def test_pow_period(self):
        op = hqml.ConditionalParity(wires=[0, 1])
        result = op.pow(4)
        assert result == []

    def test_pow_odd(self):
        op = hqml.ConditionalParity(wires=[0, 1])
        result = op.pow(1)
        assert len(result) == 1
        assert isinstance(result[0], hqml.ConditionalRotation)
        assert result[0].parameters[0] == pytest.approx(math.pi)

    def test_label(self):
        op = hqml.ConditionalParity(wires=[0, 1])
        assert op.label() == "CΠ"

    def test_fock_matrix(self):
        op = hqml.ConditionalParity(wires=[0, 1])
        dim = 3
        matrix = op.fock_matrix({0: 2, 1: dim})
        f = hqml.Fourier.compute_fock_matrix((dim,))
        fd = hqml.math.dag(f)
        expected = hqml.math.block_diag([f, fd])
        assert hqml.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)
