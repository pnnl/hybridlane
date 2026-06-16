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
from ..wires import BasisMap, ComputationalBasis
from .base import (
    CountsResult,
    SampleMeasurement,
    SampleResult,
    StateMeasurement,
)
from .probability import ProbabilityMP
from .sample import SampleMP


def expval(op: Operator | MeasurementValue) -> "ExpectationMP":
    """Expectation value of the supplied observable"""

    if isinstance(op, MeasurementValue):
        raise NotImplementedError("Mid-circuit measurement is not currently supported")

    with qp.QueuingManager.stop_recording():
        mp = qp.expval(op)

    return ExpectationMP(obs=mp.obs)


class ExpectationMP(SampleMeasurement, StateMeasurement):
    _shortname = "expval"

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
        with qp.QueuingManager.stop_recording():
            eigvals = SampleMP(
                self.obs, bases=None, eigvals=self._eigvals
            ).process_samples(
                samples,
                wire_order=self.obs.wires,
                shot_range=shot_range,
                bin_size=bin_size,
            )

        if isinstance(eigvals, list):
            return [math.mean(t) for t in eigvals]

        return math.mean(eigvals)

    def process_counts(self, counts: CountsResult) -> TensorLike:
        if counts.is_basis_states:
            with qp.QueuingManager.stop_recording():
                counts = SampleMP(
                    self.obs,
                    bases=None,
                    eigvals=self._eigvals,
                ).process_counts(counts)

        eigvals, occurences = list(zip(*counts.counts.items()))
        eigvals = math.array(eigvals)
        occurences = math.array(occurences)
        p = occurences / counts.shots

        return math.dot(eigvals, p)

    def process_state(
        self,
        state: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        if eigvals is None:
            raise ValueError(
                "Eigenvalues must be provided to compute the expectation value from the state"
            )

        eigvals = math.cast(eigvals, "float64")
        with qp.QueuingManager.stop_recording():
            # The schema here doesn't matter, we just need it to take up the full
            # state space so that the probabilities are computed correctly.
            schema = BasisMap({wire_order: ComputationalBasis.Discrete})
            probs = ProbabilityMP(bases=schema).process_state(
                state, wire_order, wire_dims
            )
        return math.dot(eigvals, probs)

    def process_density_matrix(
        self,
        dm: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        if eigvals is None:
            raise ValueError(
                "Eigenvalues must be provided to compute the expectation value from the state"
            )

        eigvals = math.cast(eigvals, "float64")
        with qp.QueuingManager.stop_recording():
            # The schema here doesn't matter, we just need it to take up the full
            # state space so that the probabilities are computed correctly.
            schema = BasisMap({wire_order: ComputationalBasis.Discrete})
            probs = ProbabilityMP(bases=schema).process_density_matrix(
                dm, wire_order, wire_dims
            )
        return math.dot(eigvals, probs)
