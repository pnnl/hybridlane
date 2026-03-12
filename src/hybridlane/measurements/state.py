# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
from collections.abc import Sequence

from pennylane.typing import TensorLike
from pennylane.wires import Wires

from hybridlane.measurements.base import StateMeasurement

from .base import Truncation


def state(wires: Wires | None = None) -> "StateMP":
    r"""State measurement process.
    Analogous to Pennylane's state measurement (:py:func:`qml.state`), but the logic of this measurement process should be handled by the device since the state is entirely dependent on the fock cutoffs set by the device.

    .. attention:: The device level implementation should preserve Pennylane's lexicographical ordering -- ie the 'top' wire is the most significant bit in the statevector.


    **Examples**
    -------
    .. code-block:: python
        :emphasize-lines: 8

        def circuit():
            hqml.FockState(  # set mode 1 to fock state 1 using qubit 0 as ancilla
                1, [0, 1]
            )
            hqml.FockState(  # set mode 2 to fock state 2 using qubit 0 as ancilla
                2, [0, 2]
            )
            return hqml.state()

    The returned statevector should correspond to the state :math:`\lvert 0,1,2 \rangle` (with cutoffs of 2,4,4)
    so the state :math:`\lvert 0,1,2 \rangle` corresponds to the index 6 in the statevector (since (0)*4*4 + (1)*4 + (2)*1 = 6).

    """

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
