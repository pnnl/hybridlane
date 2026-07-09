# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pytest
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane.measurements import DensityMatrixMP, StateMP


class TestState:
    def test_init(self):
        mp = hl.state()
        assert isinstance(mp, StateMP)
        assert mp.wires == Wires([])

    def test_process_density_matrix(self):
        mp = hl.state()
        rho = [[0.5, 0], [0, 0.5]]
        with pytest.raises(ValueError):
            mp.process_density_matrix(rho, Wires([]), {})

    @pytest.mark.all_interfaces
    def test_process_state_with_more_wires(self, like):
        # Add a qumode and check the statevector was expanded appropriately
        def f(state):
            mp = StateMP(wires=Wires([0, 1, 2]))
            new_state = mp.process_state(
                state, wire_order=Wires([0, 1]), wire_dims={0: 2, 1: 2, 2: 3}
            )
            return new_state

        if like == "jax":
            import jax

            f = jax.jit(f)  # ty:ignore[invalid-assignment]

        # Make a 2 qubit state
        state = hl.math.ones(4, like=like)
        new_state = f(state)
        expected = hl.math.kron(state, hl.math.array([1, 0, 0], like=like))
        assert new_state == pytest.approx(expected)

    def test_process_state_with_fewer_wires(self):
        # Make a 3 qubit state
        state = hl.math.ones(8) / (2 + 0j)

        # Now try to process to a 2 qubit state
        mp = StateMP(wires=Wires([0, 1]))
        with pytest.raises(ValueError):
            mp.process_state(
                state, wire_order=Wires([0, 1, 2]), wire_dims={0: 2, 1: 2, 2: 2}
            )


class TestDensityMatrix:
    def test_init(self):
        mp = hl.density_matrix()
        assert isinstance(mp, DensityMatrixMP)
        assert mp.wires == Wires([])

        mp = hl.density_matrix(wires=(0, 1))
        assert mp.wires == Wires([0, 1])

    @pytest.mark.all_interfaces
    def test_process_state_with_fewer_wires(self, like):
        mp = hl.density_matrix(wires=0)

        def f(state):
            return mp.process_state(
                state, wire_order=Wires([0, 1]), wire_dims={0: 2, 1: 2}
            )

        if like == "jax":
            import jax

            f = jax.jit(f)  # ty:ignore[invalid-assignment]

        state = hl.math.array([1, 0, 1, 0], like=like) / hl.math.sqrt(2)
        expected = hl.math.array([[0.5, 0.5], [0.5, 0.5]], like=like)
        assert f(state) == pytest.approx(expected)

    @pytest.mark.all_interfaces
    def test_process_state_with_more_wires(self, like):
        mp = hl.density_matrix(wires=(0, 1, 2))

        def f(state):
            return mp.process_state(
                state, wire_order=Wires([0, 1]), wire_dims={0: 2, 1: 2, 2: 3}
            )

        if like == "jax":
            import jax

            f = jax.jit(f)  # ty:ignore[invalid-assignment]

        state = hl.math.array([1, 0, 1, 0], like=like) / hl.math.sqrt(2)
        rho01 = hl.math.outer(state, state)
        expected = hl.math.kron(
            rho01, hl.math.diag(hl.math.array([1, 0, 0], like=like))
        )
        assert f(state) == pytest.approx(expected)

    @pytest.mark.all_interfaces
    def test_process_density_matrix_with_more_wires(self, like):
        # Add a qumode and check the density matrix was expanded appropriately
        def f(rho):
            mp = DensityMatrixMP(wires=Wires([0, 1, 2]))
            new_rho = mp.process_density_matrix(
                rho, wire_order=Wires([0, 1]), wire_dims={0: 2, 1: 2, 2: 3}
            )
            return new_rho

        if like == "jax":
            import jax

            f = jax.jit(f)  # ty:ignore[invalid-assignment]

        # Make a 2 qubit density matrix
        rho = hl.math.ones((4, 4), like=like) / 4
        new_rho = f(rho)
        expected = hl.math.kron(rho, hl.math.diag(hl.math.array([1, 0, 0], like=like)))
        assert new_rho == pytest.approx(expected)

    @pytest.mark.all_interfaces
    def test_process_density_matrix_with_fewer_wires(self, like):
        def f(rho):
            mp = hl.density_matrix(wires=0)
            new_rho = mp.process_density_matrix(
                rho, wire_order=Wires([0, 1, 2]), wire_dims={0: 2, 1: 2, 2: 3}
            )
            return new_rho

        if like == "jax":
            import jax

            f = jax.jit(f)  # ty:ignore[invalid-assignment]

        # Make a multi-qubit and qumode density matrix. This starting state is the
        # uniform superposition over all states, so tracing out wires 1 and 2 will leave
        # a pure state on wire 0.
        rho = hl.math.ones((12, 12), like=like) / 12
        new_rho = f(rho)
        expected = hl.math.array([[0.5, 0.5], [0.5, 0.5]], like=like)
        assert new_rho == pytest.approx(expected)
