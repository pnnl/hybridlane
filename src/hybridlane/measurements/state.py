from collections.abc import Sequence

from pennylane.typing import TensorLike
from pennylane.wires import Wires

from hybridlane.measurements.base import StateMeasurement

from .base import Truncation


def state(wires: Wires | None = None) -> "StateMP":
    """State measurement process."""

    return StateMP(wires=wires)


class StateMP(StateMeasurement):
    _shortname = "state"

    def __init__(self, wires: Wires | None = None, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @property
    def numeric_type(self):
        return complex

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        return ()  ## not as simple as 2**num_device_wires, since the wires aren't fixed dimension...

    def process_state(
        self, state: Sequence[complex], wire_order: Wires, truncation: Truncation
    ) -> TensorLike:
        # todo:
        raise NotImplementedError(
            "Currently, computing the analytic statevector should be handled by the device"
        )
