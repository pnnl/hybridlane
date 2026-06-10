# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np
import pennylane as qp
import pytest
from pennylane.exceptions import DeviceError

import hybridlane as hl
from hybridlane.devices import DefaultHybrid
from hybridlane.devices.default_hybrid.device import is_analytic_observable_supported


@pytest.mark.unit
def test_importable():
    dev = qp.device("default.hybrid")
    assert isinstance(dev, DefaultHybrid)


@pytest.mark.unit
class TestDevice:
    def test_only_one_truncation_argument_provided(self):
        # `fock_level` or `wire_dims` must be provided
        dev = qp.device("default.hybrid")
        with pytest.raises(
            DeviceError,
            match="Exactly one of 'wire_dims' or 'fock_level' must be specified",
        ):
            dev.setup_execution_config()

        # Providing both is a problem
        dev = qp.device("default.hybrid", fock_level=8, wire_dims={2: 2, 3: 3})
        with pytest.raises(
            DeviceError,
            match="Exactly one of 'wire_dims' or 'fock_level' must be specified",
        ):
            dev.setup_execution_config()

    def test_supports_analytic_observable(self):
        # Various symbolic observables
        assert all(
            map(
                is_analytic_observable_supported,
                [
                    hl.X(0),
                    hl.P(0),
                    hl.N(0),
                    0.5 * hl.N(0),
                    0.123 * qp.X(0) + 0.456 * hl.P(1),
                    hl.X(0) @ hl.P(0) + hl.P(0) @ hl.X(0),
                ],
            )
        )

        # Mid-circuit measurement values
        assert is_analytic_observable_supported(qp.measure(0))

        # Matrix-based observables
        obs = qp.Hermitian(np.array([[0, 1], [1, 0]]), wires=0)
        assert is_analytic_observable_supported(obs)

    # todo: remove in v0.9.0
    def test_operator_batching_isnt_supported(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @qp.qnode(dev)
        def circuit():
            hl.Displacement([0.1, 0.2], 0, wires=0)
            return hl.expval(hl.X(0))

        with pytest.raises(DeviceError, match="Operator batching is not supported"):
            circuit()

    # todo: remove in v0.9.0
    def test_mcm_isnt_supported(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @qp.set_shots(10)
        @qp.qnode(dev, mcm_method="one-shot")
        def circuit():
            qp.H(0)
            mv = qp.measure(0)
            qp.cond(mv, hl.D)(0.123, 0, wires=1)
            return hl.expval(hl.N(1))

        with pytest.raises(DeviceError):
            circuit()
