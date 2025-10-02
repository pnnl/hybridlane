# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

r"""Contains drawing utilities for quantum circuits on the ion trap"""

from .device import _get_allowed_device_wires, QscoutIonTrap

_mode_colors = {
    0: "tomato",
    1: "orange",
    2: "gold",
    3: "lime",
    4: "turquoise",
    5: "violet",
}


def get_default_style():
    r"""Gives some defaults for drawing circuits using ``hqml.draw_mpl``

    This adds the following styles to a quantum circuit:

    * Qubits are listed before qumodes, and qumodes are plotted from low to high (in terms of mode)
    * Qumodes are colored by their mode to be rainbow

    This only works if drawing a circuit at the device level, after the circuit wires
    have been mapped to the hardware wires of the ``QscoutIonTrap`` device.
    """
    wire_order = _get_allowed_device_wires(QscoutIonTrap._max_qubits, True)

    # Color the qumodes rainbow like the slides
    icon_colors = {}
    for wire in wire_order:
        if isinstance(wire, str):
            mode = int(wire[-1])
            icon_colors[wire] = _mode_colors[mode]

    return {"wire_icon_colors": icon_colors, "wire_order": wire_order}
