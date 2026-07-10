# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Implementation of finite-shot measurements in Fock space"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pennylane as qp
from pennylane.devices.qubit.sampling import jax_random_split
from pennylane.measurements.shots import Shots
from pennylane.ops.op_math.linear_combination import LinearCombination
from pennylane.ops.op_math.sum import Sum
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ... import math
from ...measurements import (
    BasisMap,
    ComputationalBasis,
    ExpectationMP,
    ProbabilityMP,
    SampleMeasurement,
    SampleMP,
    SampleResult,
)
from .apply_operation import apply_operation
from .measure import diagonalize, flatten_state

if TYPE_CHECKING:
    from jax import Array


def measure_with_shots(
    measurements: Sequence[SampleMeasurement],
    state: TensorLike,
    shots: Shots,
    is_state_batched: bool,
    rng: Any | None = None,
    prng_key: "Array | None" = None,
    wire_map: dict | None = None,
    mid_measurements: dict | None = None,  # noqa: ARG001
) -> list[TensorLike]:
    r"""Perform finite-shot measurements on a given state."""
    results = []
    for mp in measurements:
        match mp:
            case ExpectationMP(obs=obs) if isinstance(obs, (Sum, LinearCombination)):
                measurement_func = measure_hamiltonian
            case _:
                measurement_func = measure_with_diagonalizing_gates

        prng_key, key = jax_random_split(prng_key)
        results.extend(
            measurement_func(
                mp,  # ty:ignore[invalid-argument-type]
                state,
                shots,
                is_state_batched,
                rng=rng,
                prng_key=key,
                wire_map=wire_map,
            )
        )

    return results


def measure_with_diagonalizing_gates(
    mp: SampleMeasurement,
    state: TensorLike,
    shots: Shots,
    is_state_batched: bool,
    rng: Any | None = None,
    prng_key: "Array | None" = None,
    wire_map: dict | None = None,
) -> list[TensorLike]:
    r"""Perform a measurement by diagonalizing the state in the basis of the measurement."""
    num_wires = math.ndim(state) - is_state_batched
    wire_order = Wires(range(num_wires))

    if mp.obs is not None:
        diagonalizing_gates = diagonalize(mp.obs, return_evs=False)
        for op in diagonalizing_gates:
            state = apply_operation(op, state, is_state_batched)

    basis_states = sample_state(state, shots, is_state_batched, mp.wires, rng, prng_key)
    data = {
        w: cast(TensorLike, arr)
        for w, arr in zip(mp.wires, math.unstack(basis_states, axis=-1), strict=True)
    }
    result = SampleResult.from_basis_states(data)
    result = mp.process_samples(result, wire_order)

    # Sampling wires, remap to original circuit wire labels
    if isinstance(mp, SampleMP) and mp.obs is None:
        assert isinstance(result, SampleResult)
        if wire_map is not None:
            rev_wire_map = {v: k for k, v in wire_map.items()}
            new_data = {rev_wire_map.get(w, w): arr for w, arr in result.data.items()}
            new_schema = BasisMap(
                {rev_wire_map.get(w, w): result.bases.get_basis(w) for w in result.bases.wires}
            )
            result = SampleResult(bases=new_schema, data=new_data)

    return [result]  # ty:ignore[invalid-return-type]


def measure_hamiltonian(
    mp: ExpectationMP,
    state: TensorLike,
    shots: Shots,
    is_state_batched: bool,
    rng: Any | None = None,
    prng_key: "Array | None" = None,
    wire_map: dict | None = None,  # noqa: ARG001
) -> list[TensorLike]:
    r"""Measure a Hamiltonian observable."""
    obs = mp.obs
    assert isinstance(obs, (Sum, LinearCombination))

    coeffs, terms = obs.terms()

    expvals = [
        c
        * measure_with_diagonalizing_gates(
            ExpectationMP(obs=op), state, shots, is_state_batched, rng, prng_key
        )[0]
        for c, op in zip(coeffs, terms, strict=True)
    ]

    return [math.sum(expvals)]


def sample_state(
    state: TensorLike,
    shots: Shots,
    is_state_batched: bool,
    wires: Wires | None = None,
    rng: Any | None = None,
    prng_key: "Array | None" = None,
) -> TensorLike:
    """Sample basis states from the given state

    This returns an array of shape (B, shots, num_wires) if the state is batched, or (shots,
    num_wires) if the state is not batched.
    """
    shape = math.shape(state)[int(is_state_batched) :]
    num_wires = len(shape)
    wire_order = Wires(range(num_wires))
    wire_dims = {w: shape[w] for w in wire_order}

    sampled_wires = wires or wire_order
    probs_shape = tuple(wire_dims[w] for w in sampled_wires)
    schema = BasisMap({sampled_wires: ComputationalBasis.Discrete})

    flat_state = flatten_state(state, is_state_batched)
    with qp.QueuingManager.stop_recording():
        probs = ProbabilityMP(bases=schema).process_state(flat_state, wire_order, wire_dims)

    # Add an artificial batch dimension of 1 that we'll take out later
    if not is_state_batched:
        probs = math.reshape(probs, (1, -1))

    if math.get_interface(state) == "jax" or prng_key is not None:
        indices = _sample_indices_jax(probs, shots, probs_shape, prng_key=prng_key, seed=rng)
    else:
        indices = _sample_indices_numpy(probs, shots, probs_shape, rng)

    # Indices has shape (B, shots, num_wires)
    if not is_state_batched:
        indices = math.squeeze(indices, axis=0)

    return indices  # ty:ignore[invalid-return-type]


def _sample_indices_jax(
    probs: TensorLike,
    shots: Shots,
    shape: tuple[int, ...],
    prng_key: "Array | None" = None,
    seed: Any | None = None,
) -> "Array":
    import jax
    from jax import numpy as jnp

    if prng_key is None:
        prng_key = jax.random.key(np.random.default_rng(seed).integers(2**32 - 1))

    # Produce a 2D tensor of shape (shots, num_wires) per batch element, for a total shape
    # of (batch_dim, shots, num_wires)
    def inner(probs, key):
        indices = jnp.arange(probs.shape[-1])

        # This is an array of shape (shots,) containing integers
        selected = jax.random.choice(key, indices, shape=(shots.total_shots,), p=probs)

        # Decode mixed-radix indices into basis states, producing a tuple of size num_wires,
        # where each element is an array of shape (shots,) containing integers.
        basis_states = jnp.unravel_index(selected, shape)
        return jnp.stack(basis_states, axis=-1)  # (shots, num_wires)

    batch_dim = math.shape(probs)[0]
    keys = jax_random_split(prng_key, batch_dim)
    return jax.vmap(inner, in_axes=(0, 0))(probs, keys)


def _sample_indices_numpy(
    probs: TensorLike, shots: Shots, shape: tuple[int, ...], rng: Any
) -> np.ndarray:
    rng = np.random.default_rng(rng)
    indices = math.arange(probs.shape[-1])  # ty:ignore[unresolved-attribute]

    result = []
    for prob in probs:  # ty:ignore[not-iterable]
        selected = rng.choice(indices, size=shots.total_shots, p=prob)
        basis_states = math.unravel_index(selected, shape)
        result.append(math.stack(basis_states, axis=-1))

    return math.stack(result, axis=0)
