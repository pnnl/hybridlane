# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

from collections.abc import Callable
from functools import partial

import pennylane as qp
import pytest

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
patches = pytest.importorskip("matplotlib.patches")

import hybridlane as hl  # noqa: E402
from hybridlane.drawer.mpldrawer import icon_face_color  # noqa: E402
from hybridlane.drawer.tape_mpl import (  # noqa: E402
    default_qubit_color,
    default_qumode_color,
)


def circuit1(n):
    qp.H(0)
    hl.R(0.5, 1)

    for i in range(n):
        hl.CD(0.5, 0, [0, 2 + i])

    return hl.expval(hl.N(n))


def circuit2():
    # This is just a circuit with all the custom conditional gates in it
    qp.H(0)
    hl.CS(0.123, 0.456, wires=(0, 1))
    hl.ModeSwap((1, 2))
    hl.CP((0, 1))
    hl.CR(0.123, wires=(0, 2))
    hl.CBS(0.123, 0.456, wires=(0, 1, 2))
    hl.CTMS(0.123, 0.456, wires=(0, 1, 2))
    hl.CSUM(0.5, wires=(0, 1, 2))

    return hl.expval(hl.N(2))


def circuit3():
    # Same thing but we're mixing string and int labels
    qp.H(0)
    hl.CS(0.123, 0.456, wires=(0, "m"))
    hl.ModeSwap(("m", 2))
    hl.CP((0, "m"))
    hl.CR(0.123, wires=(0, 2))
    hl.CBS(0.123, 0.456, wires=(0, "m", 2))
    hl.CTMS(0.123, 0.456, wires=(0, "m", 2))
    hl.CSUM(0.5, wires=(0, "m", 2))

    return hl.expval(hl.N(2))


@pytest.mark.unit
@pytest.mark.parametrize("f", [partial(circuit1, 3), circuit2, circuit3])
def test_draw_mpl_doesnt_error(f: Callable):
    # Test with callable invocation
    _, ax = hl.draw_mpl(f)()
    plt.close("all")

    # Test with qnode input
    dev = qp.device("default.hybrid", fock_level=8)
    qnode = qp.QNode(f, dev)
    _, ax = hl.draw_mpl(qnode)()
    plt.close("all")


@pytest.mark.unit
def test_icon_colors():
    icon_colors = {
        2: "tomato",
        3: "orange",
        4: "gold",
        5: "lime",
        6: "turquoise",
    }

    dev = qp.device("default.hybrid", fock_level=8)
    for input in (circuit1, qp.QNode(circuit1, dev)):
        _, ax = hl.draw_mpl(
            input,
            wire_icon_colors=icon_colors,
        )(5)

        icon_patches = ax.patches[-7:]

        # Check the qubit wire 0
        assert isinstance(icon_patches[0], patches.Circle)
        assert icon_patches[0].get_facecolor()[:3] == icon_face_color(
            default_qubit_color, 0.3
        )

        # Check qumode wire 1
        assert isinstance(icon_patches[1], patches.Rectangle)
        assert icon_patches[1].get_facecolor()[:3] == icon_face_color(
            default_qumode_color, 0.3
        )

        # Check the customized qumodes
        for i in range(2, 7):
            assert isinstance(icon_patches[i], patches.Rectangle)
            assert icon_patches[i].get_facecolor()[:3] == icon_face_color(
                icon_colors[i], 0.3
            )

        plt.close("all")


@pytest.mark.unit
@pytest.mark.parametrize(
    "f,wire_order", [(circuit2, (1, 0, 2)), (circuit3, ("m", 0, 2))]
)
def test_overlapping_qcond_warns(f, wire_order):
    dev = qp.device("default.hybrid", fock_level=8)
    for input in (f, qp.QNode(f, dev)):
        # put the qubit between the qumodes so that the BS gate is drawn overtop the qubit wire
        with pytest.warns(UserWarning):
            _, ax = hl.draw_mpl(input, wire_order=wire_order)()
