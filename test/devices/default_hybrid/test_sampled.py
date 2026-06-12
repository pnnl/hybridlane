# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import numpy as np
import pytest
from pennylane.measurements.shots import Shots
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane.devices.default_hybrid.sampled import sample_state


def _rng(like, seed=0):
    """Return the appropriate randomness kwargs for sample_state given an interface."""
    if like == "jax":
        import jax

        return {"prng_key": jax.random.key(seed)}
    return {"rng": seed}


@pytest.mark.unit
@pytest.mark.all_interfaces
class TestSampleState:
    """Unit tests for sample_state.

    sample_state takes a statevector and returns sampled basis states:
      - (shots, num_wires)        when is_state_batched=False
      - (batch, shots, num_wires) when is_state_batched=True
    """

    # ------------------------------------------------------------------
    # Output shape
    # ------------------------------------------------------------------

    def test_output_shape_single_wire(self, like):
        """Single qumode of dim 4, 10 shots => (10, 1)."""
        state = hl.math.array([1.0, 0.0, 0.0, 0.0], like=like)
        result = sample_state(state, Shots(10), is_state_batched=False, **_rng(like))
        assert result.shape == (10, 1)

    def test_output_shape_multi_wire(self, like):
        """Two wires (qumode dim 4, qubit dim 2), 6 shots => (6, 2)."""
        state = hl.math.zeros((4, 2), like=like)
        state = hl.math.scatter_element_add(state, [0, 0], 1.0)
        result = sample_state(state, Shots(6), is_state_batched=False, **_rng(like))
        assert result.shape == (6, 2)

    def test_output_shape_batched(self, like):
        """Two batch elements, single wire dim 4, 5 shots => (2, 5, 1)."""
        state = hl.math.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], like=like
        )
        result = sample_state(state, Shots(5), is_state_batched=True, **_rng(like))
        assert result.shape == (2, 5, 1)

    # ------------------------------------------------------------------
    # Deterministic states (prob=1 on a single basis state)
    # ------------------------------------------------------------------

    def test_fock_vacuum_always_zero(self, like):
        """Vacuum state |0> must always sample Fock number 0."""
        state = hl.math.array([1.0, 0.0, 0.0, 0.0], like=like)
        result = sample_state(state, Shots(20), is_state_batched=False, **_rng(like))
        assert hl.math.all(result[:, 0] == 0)

    def test_fock_excited_state(self, like):
        """Fock state |2> must always sample Fock number 2."""
        state = hl.math.array([0.0, 0.0, 1.0, 0.0], like=like)
        result = sample_state(state, Shots(20), is_state_batched=False, **_rng(like))
        assert hl.math.all(result[:, 0] == 2)

    def test_two_wire_product_state(self, like):
        """Product state |2>|1> must return wire0=2, wire1=1 on every shot."""
        state = hl.math.zeros((4, 2), like=like)
        state = hl.math.scatter_element_add(state, [2, 1], 1.0)
        result = sample_state(state, Shots(10), is_state_batched=False, **_rng(like))
        assert hl.math.all(result[:, 0] == 2)
        assert hl.math.all(result[:, 1] == 1)

    def test_batched_deterministic(self, like):
        """Batched state: element 0 always samples 0, element 1 always samples 2."""
        state = hl.math.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], like=like
        )
        result = sample_state(state, Shots(8), is_state_batched=True, **_rng(like))
        assert hl.math.all(result[0, :, 0] == 0)
        assert hl.math.all(result[1, :, 0] == 2)

    # ------------------------------------------------------------------
    # Wire subsetting
    # ------------------------------------------------------------------

    def test_subset_wire0_from_two(self, like):
        """Sampling wire 0 from |2>|1> always gives 2, with shape (shots, 1)."""
        state = hl.math.zeros((4, 2), like=like)
        state = hl.math.scatter_element_add(state, [2, 1], 1.0)
        result = sample_state(
            state, Shots(6), is_state_batched=False, wires=Wires([0]), **_rng(like)
        )
        assert result.shape == (6, 1)
        assert hl.math.all(result[:, 0] == 2)

    def test_subset_wire1_from_two(self, like):
        """Sampling wire 1 from |2>|1> always gives 1, with shape (shots, 1)."""
        state = hl.math.zeros((4, 2), like=like)
        state = hl.math.scatter_element_add(state, [2, 1], 1.0)
        result = sample_state(
            state, Shots(6), is_state_batched=False, wires=Wires([1]), **_rng(like)
        )
        assert result.shape == (6, 1)
        assert hl.math.all(result[:, 0] == 1)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    def test_same_seed_same_result(self, like):
        """Two calls with the same seed produce identical output."""
        state = hl.math.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0.0, 0.0], like=like)
        r1 = sample_state(state, Shots(30), is_state_batched=False, **_rng(like, seed=42))
        r2 = sample_state(state, Shots(30), is_state_batched=False, **_rng(like, seed=42))
        assert hl.math.allclose(r1, r2)

    # ------------------------------------------------------------------
    # Interface preservation
    # ------------------------------------------------------------------

    def test_output_interface_matches_input(self, like):
        """The result array must have the same interface as the input state."""
        state = hl.math.array([1.0, 0.0, 0.0, 0.0], like=like)
        result = sample_state(state, Shots(5), is_state_batched=False, **_rng(like))
        assert hl.math.get_interface(result) == like

    # ------------------------------------------------------------------
    # Statistical correctness
    # ------------------------------------------------------------------

    def test_equal_superposition_qubit(self, like):
        """A 50/50 superposition on a qubit samples 0 and 1 roughly equally."""
        state = hl.math.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], like=like)
        shots = 10_000
        result = sample_state(state, Shots(shots), is_state_batched=False, **_rng(like))
        frac_zero = hl.math.sum(result[:, 0] == 0) / shots
        assert float(frac_zero) == pytest.approx(0.5, abs=0.02)
