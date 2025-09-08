# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from unittest.mock import Mock, patch

import numpy as np
import pennylane as qml
import pytest
from pennylane.measurements import SampleMP as OldSampleMP
from pennylane.wires import Wires

from hybridlane.measurements.base import (
    SampleResult,
)
from hybridlane.sa.base import BasisSchema, ComputationalBasis
from hybridlane.measurements.sample import SampleMP
from hybridlane.ops.cv import NumberOperator, QuadX


class TestSampleMP:
    """Unit tests for the SampleMP class."""

    def test_shape(self):
        """Test the shape method."""
        mp = SampleMP(qml.Identity(0))
        assert mp.shape(shots=100) == (100,)

    def test_process_samples_pass_through(self):
        """Test that process_samples passes through basis states if obs is None."""
        mp = SampleMP(
            obs=None,
            schema=BasisSchema(
                {
                    Wires(0): ComputationalBasis.Position,
                    Wires(1): ComputationalBasis.Discrete,
                }
            ),
        )
        samples = Mock()
        assert mp.process_samples(samples, wire_order=Wires([0, 1])) is samples

    def test_process_samples_value_error(self):
        """Test that process_samples raises ValueError if eigenvalues are provided."""
        mp = SampleMP(obs=qml.PauliX(0))
        samples = Mock(is_basis_states=False)
        with pytest.raises(ValueError, match="Already provided eigenvalues"):
            mp.process_samples(samples, wire_order=Wires(0))

    @pytest.mark.parametrize(
        "obs, expected",
        [
            (qml.PauliX(0), False),
            (QuadX(0), True),
            (qml.prod(QuadX(0), qml.PauliZ(1)), True),
            (qml.prod(qml.PauliX(0), qml.PauliZ(1)), False),
            (NumberOperator(0), True),
        ],
    )
    def test_has_spectral_part(self, obs, expected):
        """Test the _has_spectral_part method."""
        mp = SampleMP(obs)
        assert mp._has_spectral_part(obs) == expected

    def test_sample_observable_prod(self):
        """Test _sample_observable with a qml.Prod."""

        prod_obs = qml.prod(qml.PauliZ(0), QuadX(1))
        mp = SampleMP(obs=prod_obs)

        z = np.array([0, 1, 1, 0])
        x = np.array([0, 1.7, -3.14, -0.0005])
        result = SampleResult(
            {0: z, 1: x},  # type: ignore
            schema=BasisSchema(
                {
                    Wires(0): ComputationalBasis.Discrete,
                    Wires(1): ComputationalBasis.Position,
                }
            ),
        )

        eigvals = mp._sample_observable(prod_obs, result)
        expected_eigvals = (1 - 2 * z) * x
        assert np.array_equal(eigvals, expected_eigvals)

    def test_sample_observable_sprod(self):
        """Test _sample_observable with a qml.SProd."""
        coeff = 2.5
        prod_obs = qml.prod(qml.PauliZ(0), QuadX(1))
        sprod_obs = qml.s_prod(coeff, prod_obs)
        mp = SampleMP(obs=sprod_obs)

        z = np.array([0, 1, 1, 0])
        x = np.array([0, 1.7, -3.14, -0.0005])
        result = SampleResult(
            {0: z, 1: x},  # type: ignore
            schema=BasisSchema(
                {
                    Wires(0): ComputationalBasis.Discrete,
                    Wires(1): ComputationalBasis.Position,
                }
            ),
        )

        eigvals = mp._sample_observable(sprod_obs, result)
        expected_eigvals = coeff * (1 - 2 * z) * x
        assert np.array_equal(eigvals, expected_eigvals)

    def test_sample_observable_pow(self):
        """Test _sample_observable with a qml.Pow."""
        power = 2
        prod_obs = qml.prod(qml.PauliZ(0), QuadX(1))
        pow_obs = qml.pow(prod_obs, power)
        mp = SampleMP(obs=pow_obs)

        z = np.array([0, 1, 1, 0])
        x = np.array([0, 1.7, -3.14, -0.0005])
        result = SampleResult(
            {0: z, 1: x},  # type: ignore
            schema=BasisSchema(
                {
                    Wires(0): ComputationalBasis.Discrete,
                    Wires(1): ComputationalBasis.Position,
                }
            ),
        )

        eigvals = mp._sample_observable(pow_obs, result)
        base_eigvals = (1 - 2 * z) * x
        expected_eigvals = base_eigvals**power
        assert np.array_equal(eigvals, expected_eigvals)

    def test_sample_observable_has_spectrum_value_error_not_diagonal(self):
        """Test _sample_observable with HasSpectrum raises ValueError if not diagonal."""

        # Use a number operator (diagonal in fock), with results that were measured
        # in the position basis
        obs = NumberOperator(0)
        mp = SampleMP(obs=obs)

        schema = BasisSchema({Wires(0): ComputationalBasis.Position})
        result = SampleResult({0: np.random.randn(5)}, schema=schema)

        with pytest.raises(ValueError, match="This observable is not diagonal"):
            mp._sample_observable(obs, result)

    def test_sample_observable_regular_operator(self):
        """Test _sample_observable with a regular PennyLane operator."""
        obs = qml.PauliX(0)
        mp = SampleMP(obs=obs)
        samples = SampleResult({0: np.array([[1], [0]])})

        with patch.object(
            OldSampleMP, "process_samples", return_value=np.array([-1, 1])
        ) as mock_process:
            eigvals = mp._sample_observable(obs, samples)
            assert np.array_equal(eigvals, np.array([-1, 1]))
            mock_process.assert_called_once()
