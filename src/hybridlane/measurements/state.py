from collections.abc import Sequence

import pennylane as qml
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from hybridlane.measurements.base import StateMeasurement

from .base import Truncation


def state() -> "StateMP":
    """State measurement process."""

    return StateMP()


class StateMP(StateMeasurement):
    _shortname = "state"

    @property
    def numeric_type(self):
        return complex

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        return ()  ## not as simple as 2**num_device_wires, since the wires aren't fixed dimension...

    def process_state(
        self, state: Sequence[complex], wire_order: Wires, truncation: Truncation
    ) -> TensorLike:
        return qml.math.array(state)
