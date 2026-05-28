# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
from collections.abc import Mapping
from typing import Hashable

import pennylane as qp
from pennylane.operation import Operator
from pennylane.ops.mid_measure import MeasurementValue
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .. import math
from ..sa import BasisSchema, ComputationalBasis
from .base import (
    CountsResult,
    SampleMeasurement,
    SampleResult,
    StateMeasurement,
)
from .probability import ProbabilityMP
from .sample import SampleMP


def var(op: Operator | MeasurementValue) -> "VarianceMP":
    """Variance of the supplied observable"""

    if isinstance(op, MeasurementValue):
        raise NotImplementedError("Mid-circuit measurement is not currently supported")

    with qp.QueuingManager.stop_recording():
        mp = qp.var(op)

    return VarianceMP(obs=mp.obs)


class VarianceMP(SampleMeasurement, StateMeasurement):
    _shortname = "var"

    @property
    def numeric_type(self):
        return float

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        return ()

    def process_samples(
        self,
        samples: SampleResult,
        wire_order: Wires | None = None,
        shot_range: tuple[int, ...] | None = None,
        bin_size: int | None = None,
    ) -> TensorLike | list[TensorLike]:
        if samples.is_basis_states:
            with qp.QueuingManager.stop_recording():
                eigvals = SampleMP(
                    self.obs, schema=None, eigvals=self._eigvals
                ).process_samples(
                    samples,
                    wire_order=self.obs.wires,
                    shot_range=shot_range,
                    bin_size=bin_size,
                )
        else:
            eigvals = samples.eigvals

        if isinstance(eigvals, list):
            return [qp.math.var(t) for t in eigvals]

        return qp.math.var(eigvals)

    def process_counts(self, counts: CountsResult) -> TensorLike:
        raise NotImplementedError()

        if counts.is_basis_states:
            with qp.QueuingManager.stop_recording():
                counts = SampleMP(
                    self.obs, schema=None, eigvals=self._eigvals
                ).process_counts(counts)

        eigvals, occurences = list(zip(*counts.counts.items()))
        eigvals = qp.math.array(eigvals)
        occurences = qp.math.array(occurences)
        p = occurences / counts.shots

        return qp.math.dot(eigvals, p)

    def process_state(
        self,
        state: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        if eigvals is None:
            raise ValueError(
                "Eigenvalues must be provided to compute the variance from the state"
            )

        eigvals = math.cast(eigvals, "float64")
        with qp.QueuingManager.stop_recording():
            # The schema here doesn't matter, we just need it to take up the full
            # state space so that the probabilities are computed correctly.
            schema = BasisSchema({wire_order: ComputationalBasis.Discrete})
            probs = ProbabilityMP(schema=schema).process_state(
                state, wire_order, wire_dims
            )
        expval2 = math.dot(eigvals**2, probs)
        expval = math.dot(eigvals, probs)
        return expval2 - expval**2

    def process_density_matrix(
        self,
        dm: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        if eigvals is None:
            raise ValueError(
                "Eigenvalues must be provided to compute the variance from the state"
            )

        eigvals = math.cast(eigvals, "float64")
        with qp.QueuingManager.stop_recording():
            # The schema here doesn't matter, we just need it to take up the full
            # state space so that the probabilities are computed correctly.
            schema = BasisSchema({wire_order: ComputationalBasis.Discrete})
            probs = ProbabilityMP(schema=schema).process_density_matrix(
                dm, wire_order, wire_dims
            )
        expval2 = math.dot(eigvals**2, probs)
        expval = math.dot(eigvals, probs)
        return expval2 - expval**2
