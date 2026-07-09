# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
# Inspired by PennyLane's default.qubit device

from functools import singledispatch
from string import ascii_letters as alphabet
from typing import cast

import numpy as np
import pennylane as qp
from pennylane.devices.qubit import apply_operation as apply_operation_qubit
from pennylane.exceptions import AdjointUndefinedError
from pennylane.operation import Operator
from pennylane.ops import Conditional, MidMeasure
from pennylane.ops.op_math.adjoint import Adjoint
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from hybridlane import math, ops


@singledispatch
def apply_operation(
    op: Operator,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **execution_kwargs,
):
    # If the operation has a matrix representation, we will need to handle it ourselves
    # because PennyLane won't be able to expand it to match the state dimensions properly.
    if op.has_matrix:
        mat = op.matrix()
        return apply_operation_einsum(mat, state, op.wires, is_state_batched)

    # Fall back to the pennylane method and then see what breaks
    return apply_operation_qubit(
        op, state, is_state_batched, debugger, **execution_kwargs
    )


@apply_operation.register
def apply_fock_operator(
    op: ops.FockRepresentation,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    """Generically applies operators that have a fock_matrix definition"""
    assert isinstance(op, Operator)

    # We store the state in shape (d_1, ..., d_n) and wires are 0..n-1 from the map_wires
    # transform, so we can always do this
    n = cast(int, math.ndim(state) - is_state_batched)
    state_shape = cast(tuple[int, ...], math.shape(state))[-n:]
    wire_dims = {w: d for w, d in enumerate(state_shape)}
    mat = op.fock_matrix(wire_dims)

    return apply_operation_einsum(mat, state, op.wires, is_state_batched)


# Need to re-implement conditional ourselves so that it calls our version of
# apply_operation and not pennylane's
@apply_operation.register
def apply_conditional(
    op: Conditional,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    mid_measurements = kwargs.get("mid_measurements")
    rng = kwargs.get("rng")
    prng_key = kwargs.get("prng_key")
    interface = math.get_interface(state)

    val = op.meas_val.concretize(mid_measurements)

    def true_fn(state):
        return apply_operation(
            op.base,
            state,
            is_state_batched=is_state_batched,
            debugger=debugger,
            rng=rng,
            prng_key=prng_key,
        )

    if interface == "jax":
        import jax

        return jax.lax.cond(
            val,
            true_fn,
            lambda x: x,  # no-op
            state,
        )

    elif val:
        return true_fn(state)

    return state


@apply_operation.register
def apply_qubit_mid_measure(op: MidMeasure, state: TensorLike, **kwargs):
    return apply_operation_qubit(op, state, **kwargs)


@apply_operation.register
def apply_qubit_conditioned(op: ops.QubitConditioned, state: TensorLike, **kwargs):
    if len(op.control_wires) != 1:
        raise NotImplementedError("Only single control wire is supported for now")

    return apply_qcond(op.base, state, op.control_wires[0], **kwargs)  # ty:ignore[invalid-argument-type]


@apply_operation.register
def apply_adjoint(
    op: Adjoint,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    try:
        with qp.QueuingManager.stop_recording():
            adj_op = op.base.adjoint()

        return apply_operation(
            adj_op,
            state,
            is_state_batched=is_state_batched,
            debugger=debugger,
            **kwargs,
        )
    except AdjointUndefinedError:
        return apply_operation_qubit(
            op,
            state,
            is_state_batched,
            debugger,
            **kwargs,
        )


######################################
#     Specialized Kernels
######################################


@apply_operation.register
def apply_optimized_pennylane_op(
    op: qp.Identity
    | qp.GlobalPhase
    | qp.X
    | qp.Y
    | qp.Z
    | qp.Hadamard
    | qp.S
    | qp.T
    | qp.RX
    | qp.RY
    | qp.RZ
    | qp.CNOT,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    """Applies common gates that Pennylane has an optimized implementation for"""
    return apply_operation_qubit(op, state, is_state_batched, debugger, **kwargs)


@apply_operation.register
def apply_number_operator(
    op: ops.N,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    dim = math.shape(state)[op.wires[0] + is_state_batched]
    diag = math.arange(dim, like=state)
    return apply_diag_operation(diag, state, op.wires, is_state_batched)


@apply_operation.register
def apply_fourier(
    op: ops.F,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    dim = math.shape(state)[op.wires[0] + is_state_batched]
    diag = math.exp(-1j * np.pi / 2 * math.arange(dim, like=state))
    return apply_diag_operation(diag, state, op.wires, is_state_batched)


@apply_operation.register
def apply_rotation(
    op: ops.R,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    dim = math.shape(state)[op.wires[0] + is_state_batched]
    diag = math.exp(-1j * op.parameters[0] * math.arange(dim, like=state))  # ty:ignore[unsupported-operator]
    return apply_diag_operation(diag, state, op.wires, is_state_batched)


@apply_operation.register
def apply_kerr(
    op: ops.K,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    dim = math.shape(state)[op.wires[0] + is_state_batched]
    diag = math.exp(-1j * op.parameters[0] * math.arange(dim, like=state) ** 2)  # ty:ignore[unsupported-operator]
    return apply_diag_operation(diag, state, op.wires, is_state_batched)


@apply_operation.register
def apply_snap(
    op: ops.SNAP,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    axis = cast(int, op.wires[0] + is_state_batched)
    slices = list(math.unstack(state, axis=axis))
    slices[op.hyperparameters["n"]] = math.multiply(
        slices[op.hyperparameters["n"]], math.exp(1j * op.parameters[0])  # ty:ignore[unsupported-operator]
    )
    return math.stack(slices, axis=axis)


@apply_operation.register
def apply_conditional_parity(
    op: ops.CP,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    cond_wire, wires = op.wires[0], op.wires[1:]
    sub_op = ops.F(wires=wires)
    return apply_qcond(sub_op, state, cond_wire, is_state_batched)


@apply_operation.register
def apply_conditional_rotation(
    op: ops.CR,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    cond_wire, wires = op.wires[0], op.wires[1:]
    sub_op = ops.R(op.parameters[0] / 2, wires=wires)  # ty:ignore[unsupported-operator]
    return apply_qcond(sub_op, state, cond_wire, is_state_batched)


@apply_operation.register
def apply_conditional_displacement(
    op: ops.CD,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    cond_wire, wires = op.wires[0], op.wires[1:]
    sub_op = ops.D(*op.parameters, wires=wires)
    return apply_qcond(sub_op, state, cond_wire, is_state_batched)


@apply_operation.register
def apply_conditional_squeezing(
    op: ops.CS,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    cond_wire, wires = op.wires[0], op.wires[1:]
    sub_op = ops.S(*op.parameters, wires=wires)
    return apply_qcond(sub_op, state, cond_wire, is_state_batched)


@apply_operation.register
def apply_conditional_sum(
    op: ops.CSUM,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    cond_wire, wires = op.wires[0], op.wires[1:]
    sub_op = ops.SUM(*op.parameters, wires=wires)
    return apply_qcond(sub_op, state, cond_wire, is_state_batched)


@apply_operation.register
def apply_conditional_tms(
    op: ops.CTMS,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    cond_wire, wires = op.wires[0], op.wires[1:]
    sub_op = ops.TMS(*op.parameters, wires=wires)
    return apply_qcond(sub_op, state, cond_wire, is_state_batched)


@apply_operation.register
def apply_conditional_bs(
    op: ops.CBS,
    state: TensorLike,
    is_state_batched: bool = False,
    debugger=None,
    **kwargs,
):
    cond_wire, wires = op.wires[0], op.wires[1:]
    sub_op = ops.BS(*op.parameters, wires=wires)
    return apply_qcond(sub_op, state, cond_wire, is_state_batched)


def apply_operation_einsum(
    mat: TensorLike,
    state: TensorLike,
    op_wires: Wires,
    is_state_batched: bool = False,
) -> TensorLike:
    # Implicit casting like in pennylane's qubit version
    mat = mat + 0j  # ty:ignore[unsupported-operator]
    n = cast(int, math.ndim(state) - is_state_batched)
    state_shape = cast(tuple[int, ...], math.shape(state))[-n:]

    m = len(op_wires)
    wire_dims = {w: d for w, d in enumerate(state_shape)}
    mat_dim = int(math.prod([wire_dims[w] for w in op_wires]))
    batch_shape = math.get_batch_size(mat, (mat_dim, mat_dim), mat_dim**2)

    state_indices = alphabet[:n]
    affected_indices = "".join(alphabet[i] for i in op_wires)
    new_indices = alphabet[n : n + m]

    new_state_indices = state_indices
    for old, new in zip(affected_indices, new_indices):
        new_state_indices = new_state_indices.replace(old, new)

    indices = (
        f"...{new_indices}{affected_indices},...{state_indices}->...{new_state_indices}"
    )

    # Reshape the matrix to have the appropriate shape for the einsum
    target_shape = tuple(wire_dims[w] for w in op_wires) * 2
    if batch_shape:
        target_shape = (*batch_shape, *target_shape)

    mat = math.reshape(mat, target_shape)
    return math.einsum(indices, mat, state)


def apply_diag_operation(
    diag: TensorLike, state: TensorLike, op_wires: Wires, is_state_batched: bool = False
) -> TensorLike:
    axis = cast(int, op_wires[0] + is_state_batched)

    slices = []
    for d, s in zip(diag, math.unstack(state, axis=axis)):  # ty:ignore[invalid-argument-type, not-iterable]
        slices.append(math.multiply(s, d))

    return math.stack(slices, axis=axis)


def apply_qcond(
    op: ops.FockRepresentation,
    state: TensorLike,
    cond_wire: int,
    is_state_batched: bool = False,
) -> TensorLike:
    assert isinstance(op, Operator)
    axis = cond_wire + is_state_batched

    # Slicing below removes the axis `cond_wire`, so we shift all wires lower to compensate
    wires = [w - 1 if w > cond_wire else w for w in op.wires]
    new_op = op.__class__(*op.parameters, wires=wires, **op.hyperparameters)

    s0, s1 = math.unstack(state, axis=axis)
    n = cast(int, math.ndim(s0) - is_state_batched)
    state_shape = cast(tuple[int, ...], math.shape(s0))[-n:]
    wire_dims = {w: d for w, d in enumerate(state_shape)}
    mat = new_op.fock_matrix(wire_dims)

    s0 = apply_operation_einsum(mat, s0, new_op.wires, is_state_batched)
    s1 = apply_operation_einsum(math.dag(mat), s1, new_op.wires, is_state_batched)

    return math.stack([s0, s1], axis=axis)
