# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qp
import pytest

import hybridlane as hl


@pytest.mark.unit
class TestConditionalParity:
    def test_init(self):
        op = hl.ConditionalParity(wires=[0, 1])
        assert op.name == "ConditionalParity"
        assert op.num_params == 0
        assert op.num_wires == 2
        assert op.parameters == []
        assert op.wires == qp.wires.Wires([0, 1])

    def test_decomposition(self):
        op = hl.ConditionalParity(wires=[0, 1])
        decomp = op.decomposition()
        assert len(decomp) == 1
        assert isinstance(decomp[0], hl.ConditionalRotation)

    def test_adjoint(self):
        op = hl.ConditionalParity(wires=[0, 1])
        adj_op = op.adjoint()
        assert isinstance(adj_op, hl.ConditionalRotation)

    def test_pow_period(self):
        op = hl.ConditionalParity(wires=[0, 1])
        result = op.pow(4)
        assert result == []

    def test_pow_odd(self):
        op = hl.ConditionalParity(wires=[0, 1])
        result = op.pow(1)
        assert len(result) == 1
        assert isinstance(result[0], hl.ConditionalRotation)
        assert result[0].parameters[0] == pytest.approx(math.pi)

    def test_label(self):
        op = hl.ConditionalParity(wires=[0, 1])
        assert op.label() == "CP"

    def test_fock_matrix(self):
        op = hl.ConditionalParity(wires=[0, 1])
        dim = 3
        matrix = op.fock_matrix({0: 2, 1: dim})
        f = hl.Fourier.compute_fock_matrix((dim,))
        fd = hl.math.dag(f)
        expected = hl.math.block_diag([f, fd])
        assert hl.math.get_interface(matrix) == "numpy"
        assert matrix == pytest.approx(expected)
