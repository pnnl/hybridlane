# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import pennylane as qp
import pytest

import hybridlane as hl


@pytest.mark.usefixtures("enable_graph_decomp")
class TestFockState:
    @pytest.mark.unit
    def test_resource_rep(self):
        op = hl.FockState(5, [0, 1])
        assert set(op.resource_params.keys()) == op.resource_keys

    @pytest.mark.integration
    @pytest.mark.bq
    @pytest.mark.parametrize("n", range(1, 10, 2))
    def test_expval(self, n):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit(n):
            hl.FockState(n, [0, 1])
            return hl.expval(qp.Z(0) @ hl.N(1))

        expval = circuit(n)
        assert np.isclose(expval, n)

    @pytest.mark.integration
    @pytest.mark.bq
    @pytest.mark.parametrize("n", range(1, 10, 2))
    def test_var(self, n):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=16)

        @qp.qnode(dev)
        def circuit(n):
            hl.FockState(n, [0, 1])
            return hl.var(qp.Z(0) @ hl.N(1))

        var = circuit(n)
        assert np.isclose(var, 0)
