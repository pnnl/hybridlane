# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pytest
from pennylane.tape import QuantumScript

import hybridlane as hl
from hybridlane.devices.default_hybrid.simulate import get_final_state
from hybridlane.devices.default_hybrid.state_prep import coherent_state


@pytest.mark.integration
class TestGetFinalState:
    @pytest.mark.all_interfaces
    def test_fock_operator(self, like):
        def f(alpha):
            tape = QuantumScript([hl.D(alpha, 0, 0)])
            return get_final_state(tape, {0: 5})

        if like == "jax":
            import jax

            f = jax.jit(f)

        alpha = hl.math.array(0.123, like=like)
        expected_state = coherent_state(alpha, 5)
        state, is_state_batched = f(alpha)

        assert not is_state_batched
        assert state.shape == (5,)
        assert hl.math.get_interface(state) == like
        assert state == pytest.approx(expected_state, abs=1e-6)
