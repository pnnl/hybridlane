# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

from pennylane.drawer import tape_mpl as pl_tape_mpl
from pennylane.drawer.tape_mpl import (
    _add_operation_to_drawer,
)
from pennylane.drawer.utils import (
    convert_wire_order,
)
from pennylane.tape import QuantumScript

from .. import ops, sa
from .mpldrawer import HybridMPLDrawer

has_mpl = True
try:
    import matplotlib as mpl
except (ModuleNotFoundError, ImportError):
    has_mpl = False

default_qumode_color = "mediumseagreen"
default_qubit_color = "darkorchid"


# Register our custom gates
# Todo: Maybe we should create a symbolicop for "conditional", not the mcm conditional
# but another kind that acts similarly to conditional displacement
@_add_operation_to_drawer.register
def _(op: ops.ConditionalDisplacement, drawer: HybridMPLDrawer, layer: int, _config):
    qubit, qumode = op.wires.tolist()
    drawer.z_conditional(layer, [qubit], wires_target=qumode, control_values=[True])

    new_op = ops.Displacement(*op.parameters, wires=qumode)
    _add_operation_to_drawer(new_op, drawer, layer, _config)


@_add_operation_to_drawer.register
def _(op: ops.ConditionalParity, drawer: HybridMPLDrawer, layer: int, _config):
    qubit, qumode = op.wires.tolist()
    drawer.z_conditional(layer, [qubit], wires_target=qumode, control_values=[True])

    new_op = ops.Fourier(wires=qumode)
    _add_operation_to_drawer(new_op, drawer, layer, _config)


@_add_operation_to_drawer.register
def _(op: ops.ConditionalRotation, drawer: HybridMPLDrawer, layer: int, _config):
    qubit, qumode = op.wires.tolist()
    drawer.z_conditional(layer, [qubit], wires_target=qumode, control_values=[True])

    new_op = ops.Rotation(*op.parameters, wires=qumode)
    _add_operation_to_drawer(new_op, drawer, layer, _config)


@_add_operation_to_drawer.register
def _(op: ops.ConditionalSqueezing, drawer: HybridMPLDrawer, layer: int, _config):
    qubit, qumode = op.wires.tolist()
    drawer.z_conditional(layer, [qubit], wires_target=qumode, control_values=[True])

    new_op = ops.Squeezing(*op.parameters, wires=qumode)
    _add_operation_to_drawer(new_op, drawer, layer, _config)


@_add_operation_to_drawer.register
def _(op: ops.ConditionalBeamsplitter, drawer: HybridMPLDrawer, layer: int, _config):
    qubit, *qumodes = op.wires.tolist()
    drawer.z_conditional(
        layer, [qubit], wires_target=min(qumodes), control_values=[True]
    )

    new_op = ops.Beamsplitter(*op.parameters, wires=qumodes)
    _add_operation_to_drawer(new_op, drawer, layer, _config)


@_add_operation_to_drawer.register
def _(
    op: ops.ConditionalTwoModeSqueezing, drawer: HybridMPLDrawer, layer: int, _config
):
    qubit, *qumodes = op.wires.tolist()
    drawer.z_conditional(
        layer, [qubit], wires_target=min(qumodes), control_values=[True]
    )

    new_op = ops.TwoModeSqueezing(*op.parameters, wires=qumodes)
    _add_operation_to_drawer(new_op, drawer, layer, _config)


@_add_operation_to_drawer.register
def _(op: ops.ConditionalTwoModeSum, drawer: HybridMPLDrawer, layer: int, _config):
    qubit, *qumodes = op.wires.tolist()
    drawer.z_conditional(
        layer, [qubit], wires_target=min(qumodes), control_values=[True]
    )

    new_op = ops.TwoModeSum(*op.parameters, wires=qumodes)
    _add_operation_to_drawer(new_op, drawer, layer, _config)


@_add_operation_to_drawer.register
def _(op: ops.ModeSwap, drawer: HybridMPLDrawer, layer: int, _config):
    drawer.SWAP(layer, op.wires.tolist())


def _draw_icons(
    drawer: HybridMPLDrawer,
    tape: QuantumScript,
    wire_order,
    show_all_wires=False,
    wire_icon_colors: dict[Any, str] | None = None,
):
    sa_res = sa.analyze(tape)

    _, wire_map = convert_wire_order(
        tape, wire_order=wire_order, show_all_wires=show_all_wires
    )

    # This section assumes the types of all wires can be inferred
    wire_colors = _get_default_colors(wire_map, sa_res) | (wire_icon_colors or {})
    icons = []
    colors = []
    for wire_label in wire_map:
        icons.append("qumode" if wire_label in sa_res.qumodes else "qubit")
        colors.append(wire_colors[wire_label])

    drawer.wire_icons(icons, colors)


def tape_mpl(
    tape: QuantumScript,
    wire_order: Sequence | None = None,
    show_all_wires: bool = False,
    show_wire_types: bool = True,
    decimals: int | None = None,
    style: str | None = None,
    *,
    fig: "mpl.figure.Figure" | None = None,
    max_length: int | None = None,
    **kwargs,
):
    """Draws a quantum circuit using matplotlib

    Args:
        tape: the operations and measurements to draw

    Keyword Args:
        wire_order: the order (from top to bottom) to print the wires of the circuit

        show_all_wires: If True, all wires, including empty wires, are printed.

        decimals: How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.

        style: Visual style of plot. See :py:func:`qml.draw_mpl <pennylane.drawer.draw.draw_mpl>`

        show_wire_types: If True, draw qumode/qubit icons next to each wire

        wire_icon_colors (dict | None): A dict of wires -> colors to use for each wire icon. If a wire
            is not provided, a default color will be used (``mediumseagreen`` for qumodes and ``darkorchid``
            for qubits). Colors are anything compatible with Matplotlib.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes._axes.Axes: the return of :py:func:`qml.draw_mpl <pennylane.drawer.draw.draw_mpl>`

    .. seealso::

        :py:func:`qml.draw_mpl <pennylane.drawer.draw.draw_mpl>`
    """
    try:
        # Patch pennylane to use our drawer instead that handles hybrid gates
        with patch("pennylane.drawer.tape_mpl.MPLDrawer", HybridMPLDrawer):
            result = pl_tape_mpl(
                tape,
                wire_order=wire_order,
                show_all_wires=show_all_wires,
                decimals=decimals,
                fig=fig,
                max_length=max_length,
                style=style,
                **kwargs,
            )

            # Now draw the icons if desired
            if show_wire_types:
                wire_icon_colors = kwargs.get("wire_icon_colors")
                for drawer in HybridMPLDrawer._instances:
                    _draw_icons(
                        drawer,
                        tape,
                        wire_order=wire_order,
                        show_all_wires=show_all_wires,
                        wire_icon_colors=wire_icon_colors,
                    )

            return result

    finally:
        HybridMPLDrawer._instances.clear()


def _get_default_colors(
    wire_map: dict[Any, int], sa_res: sa.StaticAnalysisResult
) -> dict[Any, str]:
    result = {}
    for wire in wire_map:
        if wire in sa_res.qubits:
            result[wire] = default_qubit_color
        elif wire in sa_res.qumodes:
            result[wire] = default_qumode_color

    return result
