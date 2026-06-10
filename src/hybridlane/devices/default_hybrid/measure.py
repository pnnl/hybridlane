# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import itertools
from collections.abc import Mapping, Sequence
from functools import reduce, singledispatch
from typing import Any, Hashable, cast

from pennylane.measurements import MeasurementProcess
from pennylane.operation import CV, Operator
from pennylane.ops import Pow, Prod, SProd, Sum
from pennylane.ops.mid_measure import MeasurementValue
from pennylane.ops.op_math.linear_combination import LinearCombination
from pennylane.ops.op_math.symbolicop import ScalarSymbolicOp
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import math
from ...measurements import (
    ComputationalBasis,
    ExpectationMP,
    StateMeasurement,
    VarianceMP,
)
from ...ops import attributes
from ...ops.mixins import FockRepresentation, Spectral
from .apply_operation import apply_operation


def flatten_state(state: TensorLike, is_state_batched: bool) -> TensorLike:
    if is_state_batched:
        batch_size = math.shape(state)[0]
        shape = (batch_size, -1)
    else:
        shape = (-1,)
    return math.reshape(state, shape)


def _expand_eigvals(
    eigvals: TensorLike,
    obs_wires: Wires,
    wire_order: Wires,
    wire_dims: Mapping[Hashable, int],
) -> TensorLike:
    parts = [eigvals]

    # First build the eigval vector in a different order. Tensoring with `ones` is equivalent
    # to kron(diag(eigvals), id), but more efficient
    for w in wire_order - obs_wires:
        parts.append(math.ones(wire_dims[w], like=eigvals))

    eigvals = reduce(math.kron, parts)

    # Then permute it to match `wire_order`. Because we did the expansion prior, this only
    # performs permutation
    curr_wire_order = obs_wires + wire_order
    return math.expand_vector(
        eigvals, curr_wire_order, wire_order=wire_order, wire_dims=wire_dims
    )


@singledispatch
def is_diagonalizable(obs: Operator) -> bool:
    """Determines if an observable can be diagonalized analytically in the Fock + Z basis"""
    return obs.has_diagonalizing_gates


@is_diagonalizable.register
def _(obs: CV) -> bool:
    assert isinstance(obs, Operator)

    return obs in attributes.diagonal_in_fock_basis or (
        isinstance(obs, Spectral)
        and obs.natural_basis == ComputationalBasis.Discrete
        and obs.has_diagonalizing_gates
    )


@is_diagonalizable.register
def _(obs: Sum | LinearCombination) -> bool:
    return len(obs.terms()) == 1


@is_diagonalizable.register
def _(obs: Prod) -> bool:
    # Only support tensor products
    return not obs.has_overlapping_wires and all(
        is_diagonalizable(op) for op in obs.operands
    )


@is_diagonalizable.register
def _(obs: ScalarSymbolicOp) -> bool:
    return is_diagonalizable(obs.base)


@singledispatch
def diagonalize(
    obs: Operator, wire_order: Wires, wire_dims: Mapping[Hashable, int]
) -> tuple[TensorLike, Sequence[Operator]]:
    """Computes the eigenvalues and diagonalizing gates for an observable

    This method fails if called on an observable that is not diagonalizable, so check
    with `is_diagonalizable` first.
    """

    ev = math.cast(obs.eigvals(), dtype="float64")
    gates = obs.diagonalizing_gates()
    ev = _expand_eigvals(ev, obs.wires, wire_order, wire_dims)
    return ev, gates


@diagonalize.register
def diagonalize_spectral(
    obs: Spectral, wire_order: Wires, wire_dims: Mapping[Hashable, int]
) -> tuple[TensorLike, Sequence[Operator]]:
    assert isinstance(obs, CV)
    assert isinstance(obs, Operator)

    obs_dims = tuple(wire_dims[w] for w in obs.wires)
    if len(obs.wires) == 1:
        indices = math.arange(obs_dims[0])
        ev = math.asarray(obs.fock_spectrum(indices), dtype="float64")
    else:
        grids = math.meshgrid(*[math.arange(d) for d in obs_dims], indexing="ij")
        flat_grids = [g.ravel() for g in grids]
        ev = math.asarray(obs.fock_spectrum(*flat_grids), dtype="float64")

    ev = _expand_eigvals(ev, obs.wires, wire_order, wire_dims)
    return ev, []


@diagonalize.register
def diagonalize_tensor_prod(
    obs: Prod, wire_order: Wires, wire_dims: Mapping[Hashable, int]
) -> tuple[TensorLike, Sequence[Operator]]:
    evs_and_gates = [diagonalize(op, op.wires, wire_dims) for op in obs.operands]
    evs, gates = zip(*evs_and_gates)
    ev = reduce(math.kron, evs)
    wires = Wires.all_wires([op.wires for op in obs.operands])
    ev = _expand_eigvals(ev, wires, wire_order, wire_dims)

    # Because it's a tensor product, this ordering doesn't matter
    gates = list(itertools.chain.from_iterable(gates))
    return ev, gates


@diagonalize.register
def diagonalize_sprod(
    obs: SProd, wire_order: Wires, wire_dims: Mapping[Hashable, int]
) -> tuple[TensorLike, Sequence[Operator]]:
    ev, gates = diagonalize(obs.base, wire_order, wire_dims)
    return math.asarray(obs.scalar, like=ev) * ev, gates


@diagonalize.register
def diagonalize_pow(
    obs: Pow, wire_order: Wires, wire_dims: Mapping[Hashable, int]
) -> tuple[TensorLike, Sequence[Operator]]:
    ev, gates = diagonalize(obs.base, wire_order, wire_dims)
    return ev**obs.z, gates


def build_fock_matrix(
    obs: Operator,
    wire_order: Wires,
    wire_dims: Mapping[Any, int],
) -> TensorLike:
    match obs:
        case Sum(operands=terms):
            mats = [build_fock_matrix(t, wire_order, wire_dims) for t in terms]
            return reduce(math.add, mats)
        case SProd(scalar=s, base=base):
            return s * build_fock_matrix(base, wire_order, wire_dims)
        case Pow(base=base, z=p):
            mat = build_fock_matrix(base, wire_order, wire_dims)
            return math.linalg.matrix_power(mat, p)
        case Prod(operands=ops):
            mats = [build_fock_matrix(op, wire_order, wire_dims) for op in ops]
            return reduce(math.matmul, mats)
        case FockRepresentation():
            return obs.fock_matrix(wire_dims, wire_order=wire_order)
        case _:
            # Pure qubit / PL observable
            return math.expand_matrix(
                obs.matrix(), obs.wires, wire_order=wire_order, wire_dims=wire_dims
            )


# ---------------------------------------------------------------------------
# Measurement dispatch
# ---------------------------------------------------------------------------


def diagonalizing_gates(
    mp: StateMeasurement, state: TensorLike, is_state_batched: bool
) -> TensorLike:
    shape = cast(tuple[int, ...], math.shape(state))
    n_wires = math.ndim(state) - is_state_batched
    wire_order = Wires(range(n_wires))
    wire_dims = {w: shape[i + is_state_batched] for i, w in enumerate(range(n_wires))}

    eigvals = None
    if mp.obs is not None:
        eigvals, gates = diagonalize(mp.obs, wire_order, wire_dims)

        for op in gates:
            state = apply_operation(op, state, is_state_batched)

    flat = flatten_state(state, is_state_batched)
    return mp.process_state(
        flat,
        wire_order=wire_order,
        wire_dims=wire_dims,  # ty:ignore[invalid-argument-type]
        eigvals=eigvals,
    )


def einsum(mp: StateMeasurement, state: TensorLike, is_state_batched: bool):
    shape = cast(tuple[int, ...], math.shape(state))
    n_wires = math.ndim(state) - is_state_batched
    wire_order = Wires(range(n_wires))
    wire_dims = {w: shape[i + is_state_batched] for i, w in enumerate(range(n_wires))}

    match mp:
        case ExpectationMP(obs=obs):
            mat = build_fock_matrix(obs, wire_order, wire_dims)
            state = flatten_state(state, is_state_batched)
            return math.real(math.expectation_value(mat, state))
        case VarianceMP(obs=obs):
            # fixme: there's a more efficient way to do this reusing intermediate
            # products of O|psi>
            mat = build_fock_matrix(obs, wire_order, wire_dims)
            mat2 = math.linalg.matrix_power(mat, 2)
            state = flatten_state(state, is_state_batched)
            return math.real(
                math.expectation_value(mat2, state)
                - math.expectation_value(mat, state) ** 2
            )

    raise NotImplementedError()  # pragma: no cover


def get_measurement_function(mp: StateMeasurement):
    obs = mp.obs
    if isinstance(obs, MeasurementValue) or obs is None or is_diagonalizable(obs):
        return diagonalizing_gates

    return einsum


def measure(
    mp: MeasurementProcess, state: TensorLike, is_state_batched: bool
) -> TensorLike:
    if isinstance(mp, StateMeasurement):
        func = get_measurement_function(mp)
        return func(mp, state, is_state_batched)

    raise NotImplementedError()
