# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

from __future__ import annotations
from unittest.mock import patch

from functools import wraps
from typing import TYPE_CHECKING, Callable, Literal, Optional, Sequence, Union

import pennylane as qml

from .tape_mpl import tape_mpl

if TYPE_CHECKING:
    from pennylane.workflow.qnode import QNode


def draw_mpl(
    qnode: Union[QNode, Callable],
    wire_order: Optional[Sequence] = None,
    show_all_wires: bool = False,
    show_wire_types: bool = True,
    decimals: Optional[int] = None,
    style: Optional[str] = None,
    *,
    max_length: Optional[int] = None,
    fig=None,
    level: Union[
        None, Literal["top", "user", "device", "gradient"], int, slice
    ] = "gradient",
    **kwargs,
):
    @wraps(qnode)
    def wrapper(*args, **kwargs_qnode):
        with patch("pennylane.drawer.draw.tape_mpl", tape_mpl):
            orig_wrapper = qml.draw_mpl(
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
