# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import numpy as np
import pennylane as qml
import pytest

import hybridlane as hqml


class TestDisplacement:
    def test_init(self):
        op = hqml.Displacement(0.5, 0, 0)
        assert op.name == "Displacement"
        assert op.parameters == [0.5, 0]
        assert op.wires == qml.wires.Wires(0)

    def test_heisenberg_rep(self):
        op = hqml.Displacement(np.sqrt(2), np.pi / 4, 0)
        S = op.heisenberg_tr(op.wires)
        assert hqml.heisenberg.is_symplectic(S)
        # Check we only have a shift, no mixing of x and p
        assert np.allclose(S[1:, 1:], np.eye(2))
        assert np.allclose(S[:, 0], [1, 1, 1])

    def test_adjoint(self):
        op = hqml.Displacement(1, 0.123, 0)
        adj_op = op.adjoint()
        assert adj_op.parameters == [-1, 0.123]

    def test_pow(self):
        op = hqml.Displacement(1, 0.123, 0)
        pow_op = op.pow(4)[0]
        assert pow_op.parameters == [4, 0.123]

    def test_simplify(self):
        op = hqml.Displacement(1, 0.123 + 2 * np.pi, 0)
        simplified_op = op.simplify()
        assert np.allclose(simplified_op.parameters, [1, 0.123])

        op = hqml.Displacement(0, 0.123, 0)
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestRotation:
    def test_init(self):
        op = hqml.Rotation(0.5, 0)
        assert op.name == "Rotation"
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires(0)

    def test_heisenberg_rep(self):
        op = hqml.Rotation(np.pi / 4, 0)
        S = op.heisenberg_tr(op.wires)
        assert hqml.heisenberg.is_symplectic(S)

        # Check no constant shift
        assert np.allclose(S[:, 0], [1, 0, 0])
        assert np.allclose(S[0, :], [1, 0, 0])

    def test_adjoint(self):
        op = hqml.Rotation(0.123, 0)
        adj_op = op.adjoint()
        assert adj_op.parameters == [-0.123]

    def test_pow(self):
        op = hqml.Rotation(1, 0)
        pow_op = op.pow(4)[0]
        assert pow_op.parameters == [4]

    def test_simplify(self):
        op = hqml.Rotation(0.123 + 2 * np.pi, 0)
        simplified_op = op.simplify()
        assert np.allclose(simplified_op.parameters, [0.123])

        op = hqml.Rotation(0, 0)
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestSqueezing:
    def test_init(self):
        op = hqml.Squeezing(0.5, 0, 0)
        assert op.name == "Squeezing"
        assert op.parameters == [0.5, 0]
        assert op.wires == qml.wires.Wires(0)

    def test_heisenberg_rep(self):
        op = hqml.Squeezing(2, 0, 0)
        S = op.heisenberg_tr(op.wires)
        assert hqml.heisenberg.is_symplectic(S)
        assert np.allclose(S, np.diag([1, np.exp(-2), np.exp(2)]))

    def test_adjoint(self):
        op = hqml.Squeezing(1, 0.123, 0)
        adj_op = op.adjoint()
        assert adj_op.parameters == [-1, 0.123]

    def test_pow(self):
        op = hqml.Squeezing(1, 0.123, 0)
        pow_op = op.pow(4)[0]
        assert pow_op.parameters == [4, 0.123]

    def test_simplify(self):
        op = hqml.Squeezing(1, 0.123 + 2 * np.pi, 0)
        simplified_op = op.simplify()
        assert np.allclose(simplified_op.parameters, [1, 0.123])

        op = hqml.Squeezing(0, 0.123, 0)
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestKerr:
    def test_init(self):
        op = hqml.Kerr(0.5, 0)
        assert op.name == "Kerr"
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires(0)

    def test_heisenberg_rep(self):
        op = hqml.Kerr(np.pi / 4, 0)

        # non-gaussian gate
        with pytest.raises(RuntimeError):
            op.heisenberg_tr(op.wires)

    def test_adjoint(self):
        op = hqml.Kerr(0.123, 0)
        adj_op = op.adjoint()
        assert adj_op.parameters == [-0.123]

    def test_pow(self):
        op = hqml.Kerr(1, 0)
        pow_op = op.pow(4)[0]
        assert pow_op.parameters == [4]

    def test_simplify(self):
        op = hqml.Kerr(0, 0)
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)


class TestCubicPhase:
    def test_init(self):
        op = hqml.CubicPhase(0.5, 0)
        assert op.name == "CubicPhase"
        assert op.parameters == [0.5]
        assert op.wires == qml.wires.Wires(0)

    def test_heisenberg_rep(self):
        op = hqml.CubicPhase(np.pi / 4, 0)

        # non-gaussian gate
        with pytest.raises(RuntimeError):
            op.heisenberg_tr(op.wires)

    def test_adjoint(self):
        op = hqml.CubicPhase(0.123, 0)
        adj_op = op.adjoint()
        assert adj_op.parameters == [-0.123]

    def test_pow(self):
        op = hqml.CubicPhase(1, 0)
        pow_op = op.pow(4)[0]
        assert pow_op.parameters == [4]

    def test_simplify(self):
        op = hqml.CubicPhase(0, 0)
        simplified_op = op.simplify()
        assert isinstance(simplified_op, qml.Identity)
