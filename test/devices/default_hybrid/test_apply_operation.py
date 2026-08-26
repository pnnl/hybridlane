# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import itertools

import numpy as np
import pennylane as qp
import pytest
from pennylane.ops import Conditional, Operation
from pennylane.ops.op_math.controlled import Controlled
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane.devices.default_hybrid.apply_operation import (
    apply_diag_operation,
    apply_fock_operator,
    apply_operation,
    apply_qubit_controlled,
)
from hybridlane.ops.mixins import FockRepresentation
from hybridlane.ops.op_math import QubitConditioned
from hybridlane.wires.type_check import infer_wires


def random_state(shape, like):
    rng = np.random.default_rng()
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    state = hl.math.asarray(real + 1j * imag, like=like, dtype="complex128")
    return state / hl.math.linalg.norm(state)


@pytest.mark.unit
class TestApplyOperation:
    def test_dispatch(self):
        generic = apply_operation.dispatch(Operation)

        for symbolic_type in (
            FockRepresentation,
            Conditional,
            Controlled,
            QubitConditioned,
        ):
            assert apply_operation.dispatch(symbolic_type) != generic

        # Test that our cv-specific kernels are different from the general einsum method
        for type in (
            hl.N,
            hl.F,
            hl.R,
            hl.K,
            hl.SNAP,
            hl.CP,
            hl.CR,
            hl.CD,
            hl.CS,
            hl.CSUM,
            hl.CTMS,
            hl.CBS,
        ):
            assert apply_operation.dispatch(type) != apply_fock_operator

        # Test CNOT uses optimized pennylane logic and not our controlled branch
        assert apply_operation.dispatch(qp.CNOT) != apply_qubit_controlled

    @pytest.mark.parametrize(
        "op",
        [
            qp.Identity(0),
            qp.X(0),
            qp.Y(0),
            qp.Z(0),
            qp.H(0),
            qp.S(0),
            qp.T(0),
            qp.RX(0.123, 0),
            qp.RY(0.123, 0),
            qp.RZ(0.123, 0),
            qp.CNOT(wires=(0, 1)),
        ],
    )
    @pytest.mark.all_interfaces
    def test_dv_kernel(self, op: Operation, like):
        params, metadata = op._flatten()
        params = hl.math.asarray(op.parameters, like=like)
        op = op._unflatten(params, metadata)

        batch_size = 3
        inner_shape = (2,) * len(op.wires)

        def f(state, is_batched):
            return apply_operation(op, state, is_state_batched=is_batched)

        if like == "jax":
            import jax

            f = jax.jit(f, static_argnums=1)  # ty:ignore[invalid-assignment]

        for is_batched in (False, True):
            shape = inner_shape if not is_batched else (batch_size, *inner_shape)
            state = random_state(shape, like=like)
            m = hl.math.asarray(op.matrix(), like=like)

            flat_state = (
                hl.math.reshape(state, (batch_size, -1))
                if is_batched
                else hl.math.reshape(state, (-1,))
            )
            expected_result = hl.math.matvec(m, flat_state) if is_batched else m @ flat_state
            expected_result = hl.math.reshape(expected_result, shape)

            result = f(state, is_batched)
            assert result == pytest.approx(expected_result)

    @pytest.mark.parametrize(
        "op",
        [
            hl.N(0),
            hl.SNAP(0.123, 1, 0),
            hl.F(0),
            hl.R(0.123, 0),
            hl.K(0.123, 0),
        ],
    )
    @pytest.mark.all_interfaces
    def test_cv_kernel(self, op, like):
        params = hl.math.asarray(op.parameters, like=like)
        op = op.__class__(*params, **op.hyperparameters, wires=op.wires)

        dim = 5

        def f(state, is_batched):
            return apply_operation(op, state, is_state_batched=is_batched)

        if like == "jax":
            import jax

            f = jax.jit(f, static_argnums=1)  # ty:ignore[invalid-assignment]

        for is_batched in (False, True):
            shape = (dim,) if not is_batched else (2, dim)
            state = random_state(shape, like=like)
            m = hl.math.asarray(op.fock_matrix({0: dim}), like=like)
            result = f(state, is_batched)
            expected_result = hl.math.matvec(m, state) if is_batched else m @ state

            assert result == pytest.approx(expected_result)

    @pytest.mark.parametrize(
        "op",
        [
            hl.CP(wires=(0, 1)),
            hl.CR(0.123, wires=(0, 1)),
            hl.CR(0.123, wires=(1, 0)),
            hl.CD(0.123, -0.456, wires=(0, 1)),
            hl.CS(0.123, -0.456, wires=(0, 1)),
            hl.CSUM(0.123, wires=(0, 1, 2)),
            hl.CTMS(0.123, -0.456, wires=(0, 1, 2)),
            hl.CBS(0.123, -0.456, wires=(0, 1, 2)),
        ],
    )
    @pytest.mark.all_interfaces
    def test_qcond_kernel(self, op, like):
        params = hl.math.asarray(op.parameters, like=like)
        op = op.__class__(*params, **op.hyperparameters, wires=op.wires)

        batch_size = 3
        dim = 5
        wire_dims = {op.wires[0]: 2} | dict.fromkeys(op.wires[1:], dim)
        inner_shape = tuple(wire_dims[wire] for wire in range(len(op.wires)))

        def f(state, is_batched):
            return apply_operation(op, state, is_state_batched=is_batched)

        if like == "jax":
            import jax

            f = jax.jit(f, static_argnums=1)  # ty:ignore[invalid-assignment]

        for is_batched in (False, True):
            shape = inner_shape if not is_batched else (batch_size, *inner_shape)
            state = random_state(shape, like=like)
            m = hl.math.asarray(
                op.fock_matrix(wire_dims, wire_order=range(len(op.wires))), like=like
            )

            flat_state = (
                hl.math.reshape(state, (batch_size, -1))
                if is_batched
                else hl.math.reshape(state, (-1,))
            )
            expected_result = hl.math.matvec(m, flat_state) if is_batched else m @ flat_state
            expected_result = hl.math.reshape(expected_result, shape)

            result = f(state, is_batched)
            assert result == pytest.approx(expected_result)

    @pytest.mark.parametrize("op,dim", [(qp.RX(0.123, wires=0), 2), (hl.R(0.123, wires=0), 5)])
    @pytest.mark.all_interfaces
    def test_conditional_kernel(self, op, dim, like):
        params = hl.math.asarray(op.parameters, like=like)
        then_op = op.__class__(*params, **op.hyperparameters, wires=op.wires)
        state = random_state(dim, like=like)

        def f(state, val):
            mv = qp.measure(wires=1)
            op = Conditional(mv, then_op)
            return apply_operation(op, state, mid_measurements={mv.measurements[0]: val})

        if like == "jax":
            import jax

            f = jax.jit(f)  # ty:ignore[invalid-assignment]

        for val in (0, 1):
            result = f(state, val)
            if val == 0:
                assert result == pytest.approx(state)
            else:
                assert result != pytest.approx(state)

    @pytest.mark.parametrize(
        "op",
        [
            qp.ctrl(hl.F(0), control=1),
            qp.ctrl(hl.D(0.5, 0, wires=0), control=[1, 2]),
            qp.ctrl(hl.BS(0.5, 0.123, wires=[0, 1]), control=[2]),
        ],
    )
    @pytest.mark.all_interfaces
    def test_controlled_kernel(self, op: Controlled, like):
        dim = 5

        # Test all wire orderings
        for wires in itertools.permutations(op.wires):
            op = op.map_wires(dict(enumerate(wires)))
            res = infer_wires(op, {})
            wire_dims = {w: 2 if res[w] == hl.Qubit() else dim for w in op.wires}
            shape = tuple(wire_dims[w] for w in range(len(op.wires)))

            # Change the control values
            for control_values in itertools.permutations(range(2), len(op.control_wires)):
                base = op.base
                params = hl.math.asarray(base.parameters, like=like)

                op = qp.ctrl(
                    base.__class__(*params, **base.hyperparameters, wires=base.wires),
                    control=op.control_wires,
                    control_values=control_values,
                )

                mat = hl.fock_matrix(op, wire_dims=wire_dims, wire_order=range(len(op.wires)))

                def f(state, is_batched):
                    return apply_operation(op, state, is_state_batched=is_batched)  # noqa: B023

                if like == "jax":
                    import jax

                    f = jax.jit(f, static_argnums=1)  # ty:ignore[invalid-assignment]

                for is_batched in (False, True):
                    batched_shape = shape if not is_batched else (3, *shape)
                    state = random_state(batched_shape, like=like)

                    flat_state = (
                        hl.math.reshape(state, (3, -1))
                        if is_batched
                        else hl.math.reshape(state, (-1,))
                    )
                    expected_result = (
                        hl.math.matvec(mat, flat_state) if is_batched else mat @ flat_state
                    )
                    expected_result = hl.math.reshape(expected_result, batched_shape)

                    result = f(state, is_batched)
                    assert result == pytest.approx(expected_result)


@pytest.mark.unit
class TestApplyDiagOperation:
    @pytest.mark.all_interfaces
    def test_diag1(self, like):
        dim = 5
        diag = hl.math.exp(1j * hl.math.arange(dim, like=like))
        state = hl.math.ones(dim, like=like, dtype="complex128")
        state = apply_diag_operation(diag, state, Wires([0]))
        assert state == pytest.approx(diag)
