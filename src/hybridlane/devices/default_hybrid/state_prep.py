# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Implementation of state preparation routines in Fock space"""

import functools
import itertools
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pennylane as qp
from pennylane.exceptions import DeviceError
from pennylane.operation import Operator, StatePrepBase
from pennylane.tape import QuantumScript
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

import hybridlane as hl
from hybridlane.devices.default_hybrid.measure import flatten_state

from ... import math

cv_state_prep_ops = {
    qp.CatState,
    qp.FockState,
    qp.GaussianState,
    qp.SqueezedState,
    qp.CoherentState,
    qp.FockStateVector,
    qp.BasisState,
    qp.QutritBasisState,
    hl.FockState,
}


def is_state_prep(op: Operator) -> bool:
    r"""Returns True if the operation is a recognized state preparation operation"""
    return isinstance(op, StatePrepBase) or type(op) in cv_state_prep_ops


def prepare_initial_state(
    tape: QuantumScript,
    wire_dims: dict[int, int],
    interface: str | None = None,
) -> tuple[TensorLike, int]:
    r"""Prepares initial state of the simulation

    This function merges all consecutive state preparation operations at the start of the
    tape and prepares the resulting state. If there are no prep operations, it prepares the
    state :math:`\ket{0}`

    This requires all state preparation operations to acting on disjoint systems.

    Returns:
        The state vector representing the initial state, and the index of the first non-state-prep
            operation in the tape
    """
    wires = sorted(tape.op_wires)
    state_preps = list(itertools.takewhile(is_state_prep, tape.operations))
    idx = len(state_preps)

    if not state_preps:
        return vacuum_state(wire_dims, wire_order=wires, like=interface), 0

    # Check all state preps are disjoint
    state_prep_wires = [op.wires for op in state_preps]
    if len(state_preps) > 1 and Wires.shared_wires(state_prep_wires):
        raise DeviceError("State preparations must be on disjoint wires.")

    state = tensor_state_prep(
        state_preps,
        interface,
        wire_dims=wire_dims,
        wire_order=wires,
    )
    return state, idx


def vacuum_state(
    wire_dims: Mapping[Any, int],
    wire_order: WiresLike | None = None,
    like: str | None = None,
) -> TensorLike:
    r"""Returns the vacuum state :math:`\ket{0}` for a given set of wires"""
    state_dim = tuple(wire_dims[cast(int, wire)] for wire in wire_order)  # ty:ignore[not-iterable]
    state = math.zeros(state_dim, dtype=complex)
    state[tuple(0 for _ in state_dim)] = 1.0
    state = math.asarray(state, like=like)
    return state


def tensor_state_prep(
    ops: Sequence[Operator],
    like: str | None,
    wire_dims: Mapping[Any, int],
    wire_order: WiresLike,
) -> TensorLike:
    r"""Prepares a tensor product of state preparation ops."""
    all_wires = Wires.all_wires([op.wires for op in ops])
    wire_order = Wires(wire_order)

    state_vectors = []
    for op in ops:
        dims = tuple(wire_dims[wire] for wire in op.wires)
        ket = state_vector(op, dims)
        if math.get_interface(ket) != like:
            ket = math.asarray(ket, like=like)
        state_vectors.append(ket)

    ket = functools.reduce(math.kron, state_vectors)
    ket = math.expand_vector(ket, all_wires, wire_order=wire_order, wire_dims=wire_dims)

    out_shape = tuple(wire_dims[cast(int, wire)] for wire in wire_order)
    if any(math.ndim(sv) > 1 for sv in state_vectors):
        out_shape = (-1, *out_shape)

    return math.reshape(ket, out_shape)


@functools.singledispatch
def state_vector(op: Operator, wire_dims: tuple[int, ...]) -> TensorLike:  # pragma: no cover
    """Returns the state vector for a preparation operation

    Args:
        op: The state preparation operation

        wire_dims: The dimensions of the wires the operation acts on, in the order of
            the operator's wires
    """
    raise NotImplementedError(f"State preparation for {type(op)} is not implemented.")


@state_vector.register
def _(op: StatePrepBase, wire_dims: tuple[int, ...]) -> TensorLike:
    ket = op.state_vector()
    dim = int(math.prod(wire_dims))
    batch_size = math.get_batch_size(ket, (dim,), dim)
    return flatten_state(ket, is_state_batched=batch_size and batch_size > 1)


@state_vector.register
def _(op: qp.CoherentState, wire_dims: tuple[int, ...]) -> TensorLike:
    a, phi = op.parameters
    alpha = a * math.exp(1j * phi)  # ty:ignore[unsupported-operator]
    return coherent_state(alpha, wire_dims[0])


@state_vector.register
def _(op: qp.CatState, wire_dims: tuple[int, ...]) -> TensorLike:
    dim = wire_dims[0]
    a, phi, p = math.broadcast_arrays(*op.parameters)

    # If there's a batch dimension, we need to reshape the last dimension to be
    # 1 so that it broadcasts with the kets below that have shape (batch_size, dim)
    if (batch_dim := math.size(a)) > 1:
        a = math.reshape(a, (batch_dim, 1))
        phi = math.reshape(phi, (batch_dim, 1))
        p = math.reshape(p, (batch_dim, 1))

    alpha = a * math.exp(1j * phi)
    plus_alpha = coherent_state(alpha, dim)
    minus_alpha = (-1) ** math.arange(dim, like=alpha) * plus_alpha
    numerator = plus_alpha + math.exp(1j * p * np.pi) * minus_alpha
    norm = math.sqrt(2 * (1 + math.cos(p * np.pi) * math.exp(-2 * a**2)))

    return numerator / norm


@state_vector.register
def _(op: qp.SqueezedState, wire_dims: tuple[int, ...]) -> TensorLike:
    raise NotImplementedError


@state_vector.register
def _(op: qp.GaussianState, wire_dims: tuple[int, ...]) -> TensorLike:
    raise NotImplementedError


@state_vector.register
def _(op: qp.FockState, wire_dims: tuple[int, ...]) -> TensorLike:
    n = op.parameters[0]
    dim = wire_dims[0]
    return fock_state(n, dim)


@state_vector.register
def _(op: hl.FockState, wire_dims: tuple[int, ...]) -> TensorLike:
    n = op.parameters[0]
    dim = wire_dims[1]
    ket = fock_state(n, dim)
    return math.kron(math.array([1, 0]), ket)


@state_vector.register
def _(op: qp.FockStateVector, wire_dims: tuple[int, ...]) -> TensorLike:
    # According to the pennylane docs, this should have shape (dim,) * num_wires, so
    # homogeneous dimension. We need to check its shape and if necessary, embed it into
    # the larger dimension we're simulating.
    state = op.parameters[0]
    shape = math.shape(state)

    # using math.get_batch_size fails because the state vector doesn't have to exactly match
    # wire_dims since we allow for embedding in higher dimensional spaces.
    batch_dims = shape[: -len(wire_dims)]
    batch_size = int(math.prod(batch_dims)) if batch_dims else None

    inner_shape = shape[1:] if batch_size is not None else shape
    if any(s > d for s, d in zip(inner_shape, wire_dims, strict=True)):
        raise ValueError(
            f"State vector has unbatched shape {inner_shape} which is incompatible with wire "
            f"dimensions {wire_dims}"
        )

    # Pad with 0 to match the target shape of wire_dims
    pad_width = [(0, d - s) for s, d in zip(inner_shape, wire_dims, strict=True)]
    if batch_size:
        pad_width = [(0, 0), *pad_width]

    ket = math.pad(state, pad_width, mode="constant", constant_values=0)
    return flatten_state(ket, is_state_batched=batch_size is not None)


def coherent_state(alpha: TensorLike, dim: int) -> TensorLike:
    r"""Returns the state vector for a coherent state :math:`\ket{\alpha}`

    Args:
        alpha: The complex amplitude of the coherent state

        dim: The dimension of the Hilbert space
    """
    batch_size = math.get_batch_size(alpha, (), 1)
    if batch_size:
        alpha = math.reshape(alpha, (batch_size, 1))

    norm = math.exp(-0.5 * math.abs(alpha) ** 2)

    n = math.arange(dim, like=alpha)
    fac_n = factorial(n)
    state = alpha**n / math.sqrt(fac_n)
    return norm * state


def fock_state(n: TensorLike, dim: int) -> TensorLike:
    r"""Returns the state vector for a Fock state :math:`\ket{n}`

    Args:
        n: The number of photons in the Fock state

        dim: The dimension of the Hilbert space
    """
    n = math.asarray(n).reshape(-1)
    batch_size = math.get_batch_size(n, (), 1) or 1

    states = []
    for ni in n:
        state = math.zeros(dim)
        state[ni] = 1
        states.append(state)

    state = math.stack(states, axis=0)
    if batch_size == 1:
        state = math.squeeze(state, axis=0)

    return state


def factorial(x: TensorLike) -> TensorLike:
    r"""Returns the factorial of x"""
    import autoray as ar

    if math.get_interface(x) == "torch":
        import torch  # ty:ignore[unresolved-import]

        return torch.exp(torch.lgamma(x + 1))  # ty:ignore[unsupported-operator]

    return ar.do("scipy.special.factorial", x)
