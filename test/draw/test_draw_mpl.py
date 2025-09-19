# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import pennylane as qml
import pytest

mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")
patches = pytest.importorskip("matplotlib.patches")

import hybridlane as hqml  # noqa: E402
from hybridlane.drawer.tape_mpl import default_qubit_color, default_qumode_color  # noqa: E402
from hybridlane.drawer.mpldrawer import icon_face_color  # noqa: E402


dev = qml.device("hybrid.bosonicqiskit")


@qml.qnode(dev)
def circuit1(n):
    qml.H(0)
    hqml.Rotation(0.5, 1)

    for i in range(n):
        hqml.ConditionalDisplacement(0.5, 0, [0, 2 + i])

    return hqml.expval(hqml.NumberOperator(n))


class TestIconBehavior:
    def test_icon_colors(self):
        icon_colors = {
            2: "tomato",
            3: "orange",
            4: "gold",
            5: "lime",
            6: "turquoise",
        }

        fig, ax = hqml.draw_mpl(
            circuit1,
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
