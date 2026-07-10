# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import itertools
from contextlib import nullcontext

import pennylane as qp
import pytest
from pennylane.exceptions import DeviceError
from pennylane.operation import Operator
from pennylane.tape.qscript import QuantumScript

import hybridlane as hl
from hybridlane.devices.default_hybrid.state_prep import (
    cv_state_prep_ops,
    factorial,
    prepare_initial_state,
    state_vector,
)


@pytest.mark.unit
class TestInitialStatePrep:
    @pytest.mark.parametrize(
        "prep,should_raise",
        [
            ([], False),
            ([qp.CatState(0.123, 0.456, 0, 0)], False),
            ([qp.CatState(0.123, 0.456, 0, 0), qp.CatState(0.123, 0.456, 1, 1)], False),
            ([qp.CatState(0.123, 0.456, 0, 0), qp.CatState(0.123, 0.456, 1, 0)], True),
        ],
    )
    def test_only_tensor_products(self, prep, should_raise):
        tape = QuantumScript(ops=prep)
        context = pytest.raises(DeviceError) if should_raise else nullcontext()

        with context:
            prepare_initial_state(tape, {0: 4, 1: 4})

    @pytest.mark.all_interfaces
    def test_batched(self, like):
        prep = [
            qp.CatState([0.123, 0.456], 0, 0, wires=0),
            qp.BasisState([0, 1], wires=(1, 2)),
        ]
        tape = QuantumScript(ops=prep)
        state, idx = prepare_initial_state(tape, {0: 4, 1: 2, 2: 2}, interface=like)
        assert hl.math.shape(state) == (2, 4, 2, 2)
        assert idx == 2


@pytest.mark.unit
def test_all_have_impls():
    unimplemented = state_vector.dispatch(Operator)

    for op_type in cv_state_prep_ops:
        assert state_vector.dispatch(op_type) is not unimplemented, (
            f"Missing state vector implementation for {op_type}"
        )


@pytest.mark.unit
class TestCatState:
    def test_parity(self):
        for p in (0, 1):
            op = qp.CatState(0.123, 0.456, p, wires=0)
            ket = state_vector(op, (4,))
            assert ket[(1 - p) :: 2] == pytest.approx(0)  # ty:ignore[invalid-argument-type, not-subscriptable]

    def test_batched_prep(self):
        # Batched alpha parameter
        op = qp.CatState([0.123, 0.456], 0, 0, wires=0)
        ket = state_vector(op, (4,))
        assert hl.math.shape(ket) == (2, 4)
        n = hl.math.arange(4)
        mean_pnr = hl.math.sum(n * hl.math.abs(ket) ** 2, axis=-1)
        assert mean_pnr[0] < mean_pnr[1]

        # Batched parity parameter
        op = qp.CatState(0.123, 0, [0, 1], wires=0)
        ket = state_vector(op, (4,))
        assert hl.math.shape(ket) == (2, 4)
        assert ket[0, 1::2] == pytest.approx(0)  # ty:ignore[invalid-argument-type, not-subscriptable]
        assert ket[1, 0::2] == pytest.approx(0)  # ty:ignore[invalid-argument-type, not-subscriptable]


@pytest.mark.unit
class TestCoherentState:
    def test_mean_photon_number(self):
        alpha = 0.123 * hl.math.exp(0.5j)
        op = qp.CoherentState(0.123, 0.5, wires=0)
        ket = state_vector(op, (8,))
        n = hl.math.arange(8)
        mean_pnr = hl.math.sum(n * hl.math.abs(ket) ** 2)
        assert mean_pnr == pytest.approx(abs(alpha) ** 2)

    def test_batched_prep(self):
        op = qp.CoherentState([0.123, 0.456], 0.5, wires=0)
        ket = state_vector(op, (4,))
        assert hl.math.shape(ket) == (2, 4)
        n = hl.math.arange(4)
        mean_pnr = hl.math.sum(n * hl.math.abs(ket) ** 2, axis=-1)
        assert mean_pnr[0] < mean_pnr[1]


@pytest.mark.unit
class TestPennyLaneFockState:
    def test_basic(self):
        op = qp.FockState(3, wires=0)
        ket = state_vector(op, (8,))
        assert hl.math.shape(ket) == (8,)
        assert ket[3] == pytest.approx(1)  # ty:ignore[invalid-argument-type, not-subscriptable]
        assert hl.math.sum(ket) == pytest.approx(1)

    def test_batched_prep(self):
        op = qp.FockState([0, 1, 2], wires=0)
        ket = state_vector(op, (4,))
        assert hl.math.shape(ket) == (3, 4)
        for i in range(3):
            assert ket[i, i] == pytest.approx(1)  # ty:ignore[invalid-argument-type, not-subscriptable]
            assert hl.math.sum(ket, axis=-1) == pytest.approx(1)


@pytest.mark.unit
class TestHybridlaneFockState:
    def test_basic(self):
        op = hl.FockState(3, wires=(0, 1))
        ket = state_vector(op, (2, 8))
        assert hl.math.shape(ket) == (16,)
        assert ket[3] == pytest.approx(1)  # ty:ignore[invalid-argument-type, not-subscriptable]
        assert hl.math.sum(ket) == pytest.approx(1)

    def test_batched_prep(self):
        op = hl.FockState([0, 1, 2], wires=(0, 1))  # ty:ignore[invalid-argument-type]
        ket = state_vector(op, (2, 4))
        assert hl.math.shape(ket) == (3, 8)
        for i in range(3):
            assert ket[i, i] == pytest.approx(1)  # ty:ignore[invalid-argument-type, not-subscriptable]
            assert hl.math.sum(ket, axis=-1) == pytest.approx(1)


@pytest.mark.unit
class TestFockStateVector:
    @pytest.mark.all_interfaces
    def test_basic(self, like):
        # Binomial codeword |0L> = (|0> + |4>) / sqrt(2)
        logical_zero = hl.math.array([1, 0, 0, 0, 1], like=like) / hl.math.sqrt(2)
        logical_one = hl.math.array([0, 0, 1.0, 0, 0], like=like)  # |1L> = |2>

        # Test that a smaller space fails
        with pytest.raises(ValueError):
            op = qp.FockStateVector(logical_zero, wires=0)
            state_vector(op, (4,))

        # Test embedding the logical states in larger hilbert spaces
        for dim1, dim2 in itertools.product(range(5, 10), repeat=2):
            state = hl.math.outer(logical_zero, logical_one)
            op = qp.FockStateVector(state, wires=(0, 1))
            ket = state_vector(op, (dim1, dim2))
            assert hl.math.shape(ket) == (dim1 * dim2,)

            target_state = hl.math.pad(
                state,
                [(0, dim1 - 5), (0, dim2 - 5)],
                mode="constant",
                constant_values=0,
            )

            assert ket == pytest.approx(target_state.flatten())

    @pytest.mark.all_interfaces
    def test_batched_prep(self, like):
        logical_zero = hl.math.array([1, 0, 0, 0, 1], like=like) / hl.math.sqrt(2)
        logical_one = hl.math.array([0, 0, 1.0, 0, 0], like=like)

        state = hl.math.outer(logical_zero, logical_one)
        batched_state = hl.math.stack([state] * 5, axis=0)
        op = qp.FockStateVector(batched_state, wires=(0, 1))
        ket = state_vector(op, (5, 5))
        assert hl.math.shape(ket) == (5, 25)

        # Test a smaller space fails
        with pytest.raises(ValueError):
            state_vector(op, (4, 4))
            state_vector(op, (4, 5))
            state_vector(op, (5, 4))


@pytest.mark.unit
@pytest.mark.all_interfaces
def test_factorial(like):
    n = hl.math.arange(5, like=like)
    result = factorial(n)
    assert result == pytest.approx([1, 1, 2, 6, 24])
