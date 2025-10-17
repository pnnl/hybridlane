import numpy as np
import pennylane as qml
import pytest

import hybridlane as hqml


class TestQubitConditioned:
    def test_overlapping_wires(self):
        with pytest.raises(ValueError):
            hqml.qcond(qml.RZ(0.5, 0), 0)


class TestDecomposition:
    def test_rz_to_isingzz(self):
        op = hqml.qcond(qml.RZ(0.5, 0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [qml.IsingZZ(0.5, [1, 0])]

    def test_d_to_cd(self):
        op = hqml.qcond(hqml.Displacement(0.1, 0.2, 0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [hqml.ConditionalDisplacement(0.1, 0.2, [1, 0])]

    def test_f_to_cp(self):
        op = hqml.qcond(hqml.Fourier(0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [hqml.ConditionalParity([1, 0])]

    def test_r_to_cr(self):
        op = hqml.qcond(hqml.Rotation(0.5, 0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [hqml.ConditionalRotation(1.0, [1, 0])]

    def test_multirz(self):
        op = hqml.qcond(qml.MultiRZ(0.5, [0, 1]), [2, 3])
        assert op.has_decomposition
        assert op.decomposition() == [qml.MultiRZ(0.5, [2, 3, 0, 1])]

    def test_identity(self):
        op = hqml.qcond(qml.Identity(0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [qml.Identity([1, 0])]

    def test_cnot_decomposition(self):
        op = hqml.qcond(hqml.Displacement(0.1, 0.2, 0), [1, 2])
        assert op.has_decomposition
        assert op.decomposition() == [
            qml.CNOT([1, 2]),
            hqml.qcond(hqml.Displacement(0.1, 0.2, 0), 2),
            qml.CNOT([1, 2]),
        ]
