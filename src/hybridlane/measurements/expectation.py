# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from collections.abc import Sequence

import pennylane as qml
from pennylane.measurements.mid_measure import MeasurementValue
from pennylane.operation import Operator
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .base import (
    CountsResult,
    SampleMeasurement,
    SampleResult,
    StateMeasurement,
    Truncation,
)
from .sample import SampleMP


def expval(op: Operator | MeasurementValue) -> "ExpectationMP":
    """Expectation value of the supplied observable"""

    if isinstance(op, MeasurementValue):
        raise NotImplementedError("Mid-circuit measurement is not currently supported")

    with qml.QueuingManager.stop_recording():
        mp = qml.expval(op)

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
        if samples.is_basis_states:
            with qml.QueuingManager.stop_recording():
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
            return [qml.math.mean(t) for t in eigvals]

        return qml.math.mean(eigvals)

    def process_counts(self, counts: CountsResult) -> TensorLike:
        if counts.is_basis_states:
            with qml.QueuingManager.stop_recording():
                counts = SampleMP(
                    self.obs,
                    schema=None,
                    eigvals=self._eigvals,
                ).process_counts(counts)

        eigvals, occurences = list(zip(*counts.counts.items()))
        eigvals = qml.math.array(eigvals)
        occurences = qml.math.array(occurences)
        p = occurences / counts.shots

        return qml.math.dot(eigvals, p)

    def process_state(
        self, state: Sequence[complex], wire_order: Wires, truncation: Truncation
    ):
        # todo:
        raise NotImplementedError(
            "Currently, computing the analytic expval should be handled by the device"
        )
