# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
from unittest.mock import Mock, patch

import numpy as np
import pennylane as qp
import pytest
from pennylane.measurements import SampleMP as OldSampleMP
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane.measurements.base import (
    SampleResult,
)
from hybridlane.measurements.sample import SampleMP
from hybridlane.ops import NumberOperator, QuadX
from hybridlane.wires import BasisMap, ComputationalBasis


@pytest.mark.unit
class TestSampleMP:
    """Unit tests for the SampleMP class."""

    def test_shape(self):
        """Test the shape method."""
        mp = SampleMP(qp.Identity(0))
        assert mp.shape(shots=100) == (100,)

    def test_process_samples_pass_through(self):
        """Test that process_samples passes through basis states if obs is None."""
        mp = SampleMP(
            obs=None,
            bases=BasisMap(
                {
                    Wires(0): ComputationalBasis.Position,
                    Wires(1): ComputationalBasis.Discrete,
                }
            ),
        )
        samples = Mock()
        assert mp.process_samples(samples, wire_order=Wires([0, 1])) is samples

    @pytest.mark.all_interfaces
    def test_sample_observable_prod(self, like):
        """Test _sample_observable with a qp.Prod."""

        prod_obs = qp.prod(qp.PauliZ(0), QuadX(1))
        mp = SampleMP(obs=prod_obs)

        z = hl.math.array([0, 1, 1, 0], like=like)
        x = hl.math.array([0, 1.7, -3.14, -0.0005], like=like)
        result = SampleResult.from_basis_states({0: z, 1: x})

        eigvals = mp._sample_observable(prod_obs, result)
        expected_eigvals = (1 - 2 * z) * x
        assert hl.math.get_interface(eigvals) == like
        assert hl.math.array_equal(eigvals, expected_eigvals)

    @pytest.mark.all_interfaces
    def test_sample_observable_sprod(self, like):
        """Test _sample_observable with a qp.SProd."""
        coeff = 2.5
        prod_obs = qp.prod(qp.PauliZ(0), QuadX(1))
        sprod_obs = qp.s_prod(coeff, prod_obs)
        mp = SampleMP(obs=sprod_obs)

        z = hl.math.array([0, 1, 1, 0], like=like)
        x = hl.math.array([0, 1.7, -3.14, -0.0005], like=like)
        result = SampleResult.from_basis_states({0: z, 1: x})

        eigvals = mp._sample_observable(sprod_obs, result)
        expected_eigvals = coeff * (1 - 2 * z) * x
        assert hl.math.get_interface(eigvals) == like
        assert hl.math.array_equal(eigvals, expected_eigvals)

    @pytest.mark.all_interfaces
    def test_sample_observable_pow(self, like):
        """Test _sample_observable with a qp.Pow."""
        power = 2
        prod_obs = qp.prod(qp.PauliZ(0), QuadX(1))
        pow_obs = qp.pow(prod_obs, power)
        mp = SampleMP(obs=pow_obs)

        z = hl.math.array([0, 1, 1, 0], like=like)
        x = hl.math.array([0, 1.7, -3.14, -0.0005], like=like)
        result = SampleResult.from_basis_states({0: z, 1: x})

        eigvals = mp._sample_observable(pow_obs, result)
        base_eigvals = (1 - 2 * z) * x
        expected_eigvals = base_eigvals**power
        assert hl.math.get_interface(eigvals) == like
        assert hl.math.array_equal(eigvals, expected_eigvals)

    def test_sample_observable_has_spectrum_value_error_not_diagonal(self):
        """Test _sample_observable with HasSpectrum raises ValueError if not diagonal."""

        # Use a number operator (diagonal in fock), with results that were measured
        # in the position basis
        obs = NumberOperator(0)
        mp = SampleMP(obs=obs)

        schema = BasisMap({Wires(0): ComputationalBasis.Position})
        result = SampleResult({0: np.random.randn(5)}, bases=schema)

        with pytest.raises(ValueError, match="This observable is not diagonal"):
            mp._sample_observable(obs, result)

    @pytest.mark.all_interfaces
    def test_sample_observable_regular_operator(self, like):
        """Test _sample_observable with a regular PennyLane operator."""
        obs = qp.PauliX(0)
        mp = SampleMP(obs=obs)
        samples = SampleResult.from_basis_states(
            {0: hl.math.array([[1], [0]], like=like)}
        )

        with patch.object(
            OldSampleMP,
            "process_samples",
            return_value=hl.math.array([-1, 1], like=like),
        ) as mock_process:
            eigvals = mp._sample_observable(obs, samples)
            assert hl.math.get_interface(eigvals) == like
            assert hl.math.array_equal(eigvals, [-1, 1])
            mock_process.assert_called_once()
