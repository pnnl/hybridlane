# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qml
import pytest

import hybridlane as hqml


@pytest.mark.unit
class TestSelectiveNumberArbitraryPhase:
    def test_init(self):
        """Test the __init__ method."""
        op = hqml.SelectiveNumberArbitraryPhase(0.5, 1, 1)
        assert op.name == "SelectiveNumberArbitraryPhase"
        assert op.num_params == 1
        assert op.num_wires == 1
        assert op.parameters == [0.5]
        assert op.hyperparameters["n"] == 1
        assert op.wires == qml.wires.Wires([1])

    def test_adjoint(self):
        """Test the adjoint method."""
        op = hqml.SelectiveNumberArbitraryPhase(0.5, 1, 0)
        adj_op = op.adjoint()
        assert isinstance(adj_op, hqml.SelectiveNumberArbitraryPhase)
        assert adj_op.parameters == [-0.5]

    def test_pow(self):
        """Test the pow method."""
        op = hqml.SelectiveNumberArbitraryPhase(0.5, 1, 0)
        pow_op = op.pow(2)
        assert isinstance(pow_op[0], hqml.SelectiveNumberArbitraryPhase)
        assert pow_op[0].parameters == [1.0]

    def test_simplify(self):
        """Test the simplify method."""
        op = hqml.SelectiveNumberArbitraryPhase(0, 1, 0)
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)

    def test_matrix(self):
        """Test the fock_matrix method."""
        op = hqml.SNAP(0.5, 1, 0)
        matrix = op.fock_matrix({0: 4})
        expected_matrix = hqml.math.diag([1, hqml.math.exp(0.5j), 1, 1])
        assert hqml.math.get_interface(matrix) == "numpy"
        assert hqml.math.allclose(matrix, expected_matrix)

    @pytest.mark.jax
    def test_matrix_jax(self):
        """Test the fock_matrix method with JAX."""
        param = hqml.math.asarray(0.5, like="jax")
        op = hqml.SNAP(param, 1, 0)
        matrix = op.fock_matrix({0: 4})
        expected_matrix = hqml.math.diag([1, hqml.math.exp(0.5j), 1, 1], like="jax")
        assert hqml.math.get_interface(matrix) == "jax"
        assert hqml.math.allclose(matrix, expected_matrix)
