# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import TYPE_CHECKING, Callable, Literal
from unittest.mock import patch

import pennylane as qp

from .tape_mpl import tape_mpl

if TYPE_CHECKING:
    from pennylane.workflow.qnode import QNode


def draw_mpl(
    qnode: QNode | Callable,
    wire_order: Sequence | None = None,
    show_all_wires: bool = False,
    show_wire_types: bool = True,
    decimals: int | None = None,
    style: str | None = None,
    *,
    max_length: int | None = None,
    fig=None,
    level: Literal["top", "user", "device", "gradient"]
    | int
    | slice
    | None = "gradient",
    **kwargs,
):
    r"""Draws a circuit using matplotlib

    Args:
        qnode: The circuit to be drawn

        wire_order: The display order (top to bottom) of wires in the circuit

        show_all_wires: Whether to show all wires or just those that are used

        show_wire_types: Whether to draw qubit/qumode icons next to each wire label

        decimals: The number of decimals to print circuit parameters with. If not provided,
            parameters won't be shown.

        style: The drawing style to use. See :py:func:`qp.draw_mpl <pennylane.draw_mpl>`.

    Keyword Args:
        wire_icon_colors (dict): A dictionary mapping wires to optional matplotlib-compatible colors.
            All wires that aren't provided will use default qubit or qumode colors.

    For other arguments, see :py:func:`qp.draw_mpl <pennylane.draw_mpl>`.

    Returns:
        A function that when called, produces the same output as :py:func:`qp.draw_mpl <pennylane.draw_mpl>`

    **Examples**

    By default, Hybridlane draws quantum circuits with wire icons and default colors.

    .. code-block:: python

        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=8)

        @qp.qnode(dev)
        def circuit(n):
            for j in range(n):
                qp.X(0)
                hl.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, 1])

            return hl.expval(hl.NumberOperator(1))

        n = 5
        hl.draw_mpl(circuit, style="sketch")(n)

    .. figure:: ../../_static/draw_mpl/ex_jc_circuit.png

    Furthermore, icon colors can be adjusted from their defaults (say to color different motional modes of an ion trap). Note
    that Hybridlane also has a special notation for "qubit-conditioned" gates.

    .. code-block:: python

        @qp.qnode(dev)
        def circuit(n):
            qp.H(0)
            hl.Rotation(0.5, 1)

            for i in range(n):
                hl.ConditionalDisplacement(0.5, 0, [0, 2 + i])

            return hl.expval(hl.NumberOperator(n))

        icon_colors = {
            2: "tomato",
            3: "orange",
            4: "gold",
            5: "lime",
            6: "turquoise",
        }

        hl.draw_mpl(circuit, wire_icon_colors=icon_colors, style="sketch")(5)

    .. figure:: ../../_static/draw_mpl/colored_circuit.png

    Finally, if you don't like pretty icons, you can disable them.

    .. code:: python

        @qp.qnode(dev)
        def circuit(n):
            qp.H(0)
            hl.Rotation(0.5, 1)

            for i in range(n):
                hl.ConditionalDisplacement(0.5, 0, [0, 2 + i])

            return hl.expval(hl.NumberOperator(n))

        hl.draw_mpl(circuit, show_wire_types=False, style="sketch")(5)

    .. figure:: ../../_static/draw_mpl/no_icons.png
    """

    @wraps(qnode)
    def wrapper(*args, **kwargs_qnode):
        with patch("pennylane.drawer.draw.tape_mpl", tape_mpl):
            orig_wrapper = qp.draw_mpl(
                qnode,
                wire_order=wire_order,
                show_all_wires=show_all_wires,
                show_wire_types=show_wire_types,
                decimals=decimals,
                style=style,
                max_length=max_length,
                fig=fig,
                level=level,
                **kwargs,
            )

            return orig_wrapper(*args, **kwargs_qnode)

    return wrapper
