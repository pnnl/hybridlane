from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

import hybridlane as hqml

folder = Path(__file__).parent

dev = qml.device("hybrid.bosonicqiskit")


def ex_jc_circuit():
    @qml.qnode(dev)
    def circuit(n):
        for j in range(n):
            qml.X(0)
            hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, 1])

        return hqml.expval(hqml.NumberOperator(1))

    n = 5
    hqml.draw_mpl(circuit, style="sketch")(n)
    plt.savefig(folder / "ex_jc_circuit.png", dpi=300)


def colored_circuit():
    @qml.qnode(dev)
    def circuit(n):
        qml.H(0)
        hqml.Rotation(0.5, 1)

        for i in range(n):
            hqml.ConditionalDisplacement(0.5, 0, [0, 2 + i])

        return hqml.expval(hqml.NumberOperator(n))

    icon_colors = {
        2: "tomato",
        3: "orange",
        4: "gold",
        5: "lime",
        6: "turquoise",
    }

    fig, ax = hqml.draw_mpl(circuit, wire_icon_colors=icon_colors, style="sketch")(5)
    plt.savefig(folder / "colored_circuit.png", dpi=300)


def no_icons():
    @qml.qnode(dev)
    def circuit(n):
        qml.H(0)
        hqml.Rotation(0.5, 1)

        for i in range(n):
            hqml.ConditionalDisplacement(0.5, 0, [0, 2 + i])

        return hqml.expval(hqml.NumberOperator(n))

    fig, ax = hqml.draw_mpl(circuit, show_wire_types=False, style="sketch")(5)
    plt.savefig(folder / "no_icons.png", dpi=300)


def banner_circuit():
    @qml.qnode(dev)
    def circuit():
        for i in range(0, 10, 2):
            qml.H(i)
            hqml.ConditionalDisplacement(0.5, 0, [i, i + 1])

        hqml.ModeSwap([1, 3])
        hqml.ModeSwap([5, 7])
        hqml.ModeSwap([3, 5])
        hqml.ConditionalBeamsplitter(0.5, 0.5, [0, 7, 9])

        return hqml.expval(qml.Z(0) @ hqml.NumberOperator(1))

    icon_colors = {
        1: "tomato",
        3: "orange",
        5: "gold",
        7: "lime",
        9: "turquoise",
    }

    fig, ax = hqml.draw_mpl(circuit, wire_icon_colors=icon_colors, style="sketch")()
    plt.savefig(folder / "banner_circuit.png", dpi=300)


if __name__ == "__main__":
    ex_jc_circuit()
    colored_circuit()
    no_icons()
    banner_circuit()
