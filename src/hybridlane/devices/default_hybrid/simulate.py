# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pennylane.logging import debug_logger
from pennylane.math import Interface
from pennylane.tape import QuantumScript
from pennylane.typing import Result, TensorLike

from hybridlane.devices.default_hybrid.sampled import measure_with_shots

from ... import math
from .apply_operation import apply_operation
from .measure import measure
from .state_prep import prepare_initial_state

if TYPE_CHECKING:
    from jax import Array

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@debug_logger
def get_final_state(
    tape: QuantumScript,
    wire_dims: dict[Any, int],
    debugger=None,
    prng_key: "Array | None" = None,
    **execution_kwargs,
) -> tuple[TensorLike, bool]:
    interface = Interface(execution_kwargs.get("interface"))

    # Obtain the wire map internally to also map the wire dimensions
    tape = tape.copy()
    wire_map = tape._get_standard_wire_map() or {}
    tape = tape.map_to_standard_wires()
    wire_dims = {wire_map.get(wire, wire): dim for wire, dim in wire_dims.items()}

    # Find all the state preparations in the tape, they should be at the start
    state, idx = prepare_initial_state(tape, wire_dims, interface.get_like())
    is_state_batched = math.ndim(state) > len(tape.op_wires)

    for op in tape.operations[idx:]:
        state = apply_operation(
            op,
            state,
            is_state_batched=is_state_batched,
            debugger=debugger,
            prng_key=prng_key,
            tape_shots=tape.shots,
            **execution_kwargs,
        )

        is_state_batched |= op.batch_size is not None

    # For any wires that were not acted on, we need to set them to |0>
    if tape.wires - tape.op_wires:
        # Flatten the statevector
        shape = math.shape(state)
        batch_size = shape[0] if is_state_batched else 1
        state = math.reshape(state, (batch_size, -1))

        state = math.expand_vector(
            state, tape.op_wires, wire_dims=wire_dims, wire_order=tape.wires
        )

        # Unflatten
        new_shape = tuple(wire_dims[w] for w in tape.wires)
        new_shape = (batch_size, *new_shape) if is_state_batched else new_shape
        state = math.reshape(state, new_shape)

    return state, is_state_batched


@debug_logger
def measure_final_state(
    state: TensorLike,
    is_state_batched: bool,
    tape: QuantumScript,
    wire_dims: dict[Any, int],
    debugger=None,
    rng: Any | None = None,
    prng_key: "Array | None" = None,
    wire_map: dict[Any, Any] | None = None,
    **execution_kwargs,
) -> Result:
    if not tape.shots:
        results = tuple(
            map(lambda m: measure(m, state, is_state_batched), tape.measurements)
        )

        if len(results) == 1:
            return results[0]

        return results

    results = measure_with_shots(
        tape.measurements,
        state,
        tape.shots,
        is_state_batched,
        rng=rng,
        prng_key=prng_key,
        wire_map=wire_map,
    )

    if len(tape.measurements) == 1:
        return results[0]

    return results


@debug_logger
def simulate(
    tape: QuantumScript,
    wire_dims: dict[Any, int],
    debugger=None,
    prng_key: "Array | None" = None,
    **execution_kwargs,
) -> Result:
    state, is_state_batched = get_final_state(
        tape, wire_dims, debugger=debugger, prng_key=prng_key, **execution_kwargs
    )

    return measure_final_state(
        state,
        is_state_batched,
        tape,
        wire_dims,
        debugger=debugger,
        prng_key=prng_key,
        **execution_kwargs,
    )
