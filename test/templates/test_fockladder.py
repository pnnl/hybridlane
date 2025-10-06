# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import numpy as np
import pennylane as qml
import pytest

import hybridlane as hqml


class TestFockLadder:
    @pytest.mark.parametrize("n", range(10))
    def test_expval(self, n):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qml.qnode(dev)
        def circuit(n):
            hqml.FockLadder(n, [0, 1])
            return hqml.expval(qml.Z(0) @ hqml.N(1))

        expval = circuit(n)
        assert np.isclose(expval, (-1) ** n * n)

    @pytest.mark.parametrize("n", range(10))
    def test_var(self, n):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qml.qnode(dev)
        def circuit(n):
            hqml.FockLadder(n, [0, 1])
            return hqml.var(qml.Z(0) @ hqml.N(1))

        var = circuit(n)
        assert np.isclose(var, 0)
