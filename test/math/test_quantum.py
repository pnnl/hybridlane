# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qp
import pytest

from hybridlane import math


class TestReduceStatevector:
    @pytest.mark.parametrize("dtype", ["complex64", "complex128"])
    @pytest.mark.all_interfaces
    def test_qubit_states(self, like, dtype):
        def f(state, indices, dims):
            return math.reduce_statevector(state, indices, dims, c_dtype=dtype)

        if like == "jax":
            import jax

            f = jax.jit(f, static_argnums=(1, 2))  # ty:ignore[invalid-assignment]

        states = [
            math.array([1, 0, 0, 0], like=like),
            math.array([1, 0, 1, 0], like=like) / math.sqrt(2),
            math.array([1, 0, 0, 1], like=like) / math.sqrt(2),
        ]

        for state in states:
            for index in (0, 1):
                expected = qp.math.reduce_statevector(
                    state, indices=(index,), c_dtype=dtype
                )
                actual = f(state, indices=(index,), dims=(2, 2))
                assert math.get_dtype_name(actual) == dtype
                assert actual == pytest.approx(expected)

    @pytest.mark.all_interfaces
    def test_cv_states(self, like):
        def f(state, indices, dims):
            return math.reduce_statevector(state, indices, dims)

        if like == "jax":
            import jax

            f = jax.jit(f, static_argnums=(1, 2))  # ty:ignore[invalid-assignment]

        state0 = math.array([0, 1, 0], like=like)
        state1 = math.array([1, 0], like=like)
        state = math.kron(state0, state1)

        expected = math.outer(state0, state0)
        actual = f(state, indices=(0,), dims=(3, 2))
        assert math.get_dtype_name(actual) == "complex128"
        assert actual == pytest.approx(expected)

        expected = math.outer(state1, state1)
        actual = f(state, indices=(1,), dims=(3, 2))
        assert math.get_dtype_name(actual) == "complex128"
        assert actual == pytest.approx(expected)

        state0 = math.array([0, 1, 0], like=like)
        state1 = math.array([1, 1], like=like) / math.sqrt(2)
        state = math.kron(state0, state1)

        expected = math.outer(state, state)
        actual = f(state, indices=(0, 1), dims=(3, 2))
        assert math.get_dtype_name(actual) == "complex128"
        assert actual == pytest.approx(expected)


class TestReduceDensityMatrix:
    @pytest.mark.parametrize("dtype", ["complex64", "complex128"])
    @pytest.mark.all_interfaces
    def test_qubit_states(self, like, dtype):
        def f(state, indices, dims):
            return math.reduce_dm(state, indices, dims, c_dtype=dtype)

        if like == "jax":
            import jax

            f = jax.jit(f, static_argnums=(1, 2))  # ty:ignore[invalid-assignment]

        states = [
            math.array(
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], like=like
            ),
            math.array(
                [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]],
                like=like,
            ),
            math.array(
                [
                    [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ],
                like=like,
            ),
        ]

        for state in states:
            for index in (0, 1):
                expected = qp.math.reduce_dm(state, indices=(index,), c_dtype=dtype)
                actual = f(state, indices=(index,), dims=(2, 2))
                assert math.get_dtype_name(actual) == dtype
                assert actual == pytest.approx(expected)

    @pytest.mark.all_interfaces
    def test_cv_states(self, like):
        def f(state, indices, dims):
            return math.reduce_dm(state, indices, dims)

        if like == "jax":
            import jax

            f = jax.jit(f, static_argnums=(1, 2))  # ty:ignore[invalid-assignment]

        state0 = math.array([0, 1, 0], like=like)
        state1 = math.array([1, 1], like=like) / math.sqrt(2)
        state = math.kron(state0, state1)
        rho = math.outer(state, state)

        expected = math.outer(state0, state0)
        actual = f(rho, indices=(0,), dims=(3, 2))
        assert math.get_dtype_name(actual) == "complex128"
        assert actual == pytest.approx(expected)

        expected = math.outer(state1, state1)
        actual = f(rho, indices=(1,), dims=(3, 2))
        assert math.get_dtype_name(actual) == "complex128"
        assert actual == pytest.approx(expected)

        actual = f(rho, indices=(0, 1), dims=(3, 2))
        assert math.get_dtype_name(actual) == "complex128"
        assert actual == pytest.approx(rho)
