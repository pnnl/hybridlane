# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
from collections.abc import Mapping
from typing import Hashable

from pennylane.operation import Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .. import math
from .base import CountsResult, SampleMeasurement, SampleResult, StateMeasurement


def probs(wires: Wires | None = None, op: Operator | None = None) -> "ProbabilityMP":
    pass


class ProbabilityMP(SampleMeasurement, StateMeasurement):
    @property
    def numeric_type(self):
        return float

    def process_samples(
        self,
        samples: SampleResult,
        wire_order: Wires,
        shot_range: tuple[int, ...] | None = None,
        bin_size: int | None = None,
    ):
        raise NotImplementedError()

    def process_counts(self, counts: CountsResult):
        raise NotImplementedError()

    def process_state(
        self,
        state: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        is_f32 = math.get_dtype_name(state) in ("float32", "complex64")
        dtype = "float32" if is_f32 else "float64"

        if not self.wires:
            probs = math.abs(state) ** 2
            return math.cast(probs, dtype)

        # Expand the state vector to include any missing wires. For wires that are to be
        # marginalized over (those in wire_order but not this object), we'll put their
        # dimensions at the end
        to_delete = wire_order - self.wires
        expanded_wires = self.wires + to_delete
        state = math.expand_vector(
            state,
            wires=wire_order,
            wire_order=expanded_wires,
            wire_dims=wire_dims,
        )

        # Reshape so each dimension is a wire
        d = int(math.prod([wire_dims[w] for w in expanded_wires]))
        inner_shape = tuple(wire_dims[w] for w in expanded_wires)
        batch_size = math.get_batch_size(state, (d,), d)
        shape = (batch_size, *inner_shape) if batch_size else inner_shape
        state = math.reshape(state, shape)

        # Do |psi|^2 to obtain probs and then marginalize over the wires that are
        # to be deleted
        probs = math.abs(state) ** 2
        axes = tuple(expanded_wires.index(w) for w in to_delete)
        probs = math.sum(probs, axis=axes)

        # Flatten back down
        new_shape = (batch_size, -1) if batch_size else (-1,)
        probs = math.reshape(probs, new_shape)

        return math.cast(probs, dtype)

    def process_density_matrix(
        self,
        dm: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        # this arcane logic comes from pennylane. because the diagonal of a density matrix
        # ρ_ii gives the probability of measuring state |i>, we form vectors of
        # sqrt(p(|i>) = ρ_ii) and then pass those to process_state, which will square them
        # again to get the original probabilities while also handling reshaping for us
        if math.ndim(dm) == 2:
            probs = math.diagonal(dm)
        else:
            states = map(math.diagonal, math.unstack(dm, axis=0))
            probs = math.stack(tuple(states), axis=0)

        probs = math.convert_like(probs, dm)
        return self.process_state(math.sqrt(probs), wire_order, wire_dims)
