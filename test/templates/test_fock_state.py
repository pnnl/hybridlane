# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import numpy as np
import pennylane as qml
import pytest

import hybridlane as hqml

qml.decomposition.enable_graph()


class TestFockState:
    def test_resource_rep(self):
        op = hqml.FockState(5, [0, 1])
        assert set(op.resource_params.keys()) == op.resource_keys

    @pytest.mark.parametrize("n", range(1, 10, 2))
    def test_expval(self, n):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qml.qnode(dev)
        def circuit(n):
            hqml.FockState(n, [0, 1])
            return hqml.expval(qml.Z(0) @ hqml.N(1))

        expval = circuit(n)
        assert np.isclose(expval, n)

    @pytest.mark.parametrize("n", range(1, 10, 2))
    def test_var(self, n):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qml.qnode(dev)
        def circuit(n):
            hqml.FockState(n, [0, 1])
            return hqml.var(qml.Z(0) @ hqml.N(1))

        var = circuit(n)
        assert np.isclose(var, 0)
