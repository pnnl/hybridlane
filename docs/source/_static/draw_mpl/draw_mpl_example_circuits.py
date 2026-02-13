from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane.ops import Operation

import hybridlane as hqml
from hybridlane.ops.mixins import Hybrid

folder = Path(__file__).parent

dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

# Enable graph decomposition for QPE example
qml.decomposition.enable_graph()


def ex_jc_circuit():
    @qml.qnode(dev)
    def circuit(n):
        for j in range(n):
            qml.X(0)
            hqml.JC(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, 1])

        return hqml.expval(hqml.N(1))

    n = 5
    hqml.draw_mpl(circuit, style="sketch")(n)
    plt.savefig(folder / "ex_jc_circuit.png", dpi=300)


def colored_circuit():
    @qml.qnode(dev)
    def circuit(n):
        qml.H(0)
        hqml.R(0.5, 1)

        for i in range(n):
            hqml.CD(0.5, 0, [0, 2 + i])

        return hqml.expval(hqml.N(n))

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
        hqml.R(0.5, 1)

        for i in range(n):
            hqml.CD(0.5, 0, [0, 2 + i])

        return hqml.expval(hqml.N(n))

    fig, ax = hqml.draw_mpl(circuit, show_wire_types=False, style="sketch")(5)
    plt.savefig(folder / "no_icons.png", dpi=300)


def banner_circuit():
    @qml.qnode(dev)
    def circuit():
        for i in range(0, 10, 2):
            qml.H(i)
            hqml.CD(0.5, 0, [i, i + 1])

        hqml.ModeSwap([1, 3])
        hqml.ModeSwap([5, 7])
        hqml.ModeSwap([3, 5])
        hqml.CBS(0.5, 0.5, [0, 7, 9])

        return hqml.expval(qml.Z(0) @ hqml.N(1))

    icon_colors = {
        1: "tomato",
        3: "orange",
        5: "gold",
        7: "lime",
        9: "turquoise",
    }

    fig, ax = hqml.draw_mpl(circuit, wire_icon_colors=icon_colors, style="sketch")()
    plt.savefig(folder / "banner_circuit.png", dpi=300)


# Quantum Phase Estimation Example
class JCEvolution(Operation, Hybrid):
    num_params = 4
    num_wires = 2
    num_qumodes = 1
    resource_keys = set()

    def __init__(self, t, omega_r=1, omega_q=-1, chi=0.1, wires=None, id=None):
        super().__init__(t, omega_r, omega_q, chi, wires=wires, id=id)

    def label(
        self,
        decimals: int | None = None,
        base_label: str | None = None,
        cache: dict | None = None,
    ) -> str:
        return super().label(decimals, base_label or "U", cache)

    @property
    def resource_params(self):
        return {}


@qml.register_resources({hqml.Rotation: 1, hqml.ConditionalRotation: 1, qml.RZ: 1})
def _jc_decomp(t, omega_r, omega_q, chi, wires, **_):
    hqml.Rotation(t * omega_r, wires[1])
    qml.RZ(-t * omega_q, wires[0])
    hqml.ConditionalRotation(-t * chi, wires)


@qml.register_resources({JCEvolution: 1})
def _jc_adjoint(t, *params, wires, **_):
    JCEvolution(-t, *params, wires=wires)


@qml.register_resources({JCEvolution: 1})
def _jc_pow(t, *params, wires, z, **_):
    JCEvolution(z * t, *params, wires=wires)


qml.add_decomps(JCEvolution, _jc_decomp)
qml.add_decomps("Adjoint(JCEvolution)", _jc_adjoint)
qml.add_decomps("Pow(JCEvolution)", _jc_pow)


def quantum_phase_estimation():
    from pennylane.templates import QuantumPhaseEstimation

    t = 1
    omega_r = 2
    omega_q = 5
    chi = 0.1

    target_wires = ("q", "m")
    U = JCEvolution(t, omega_r=omega_r, omega_q=omega_q, chi=chi, wires=target_wires)

    @qml.transforms.decompose(
        gate_set={
            hqml.CR,
            hqml.R,
            hqml.Red,
            hqml.Blue,
            qml.RZ,
            qml.CRZ,
            qml.CNOT,
            qml.H,
            qml.ControlledPhaseShift,
        },
    )
    @qml.qnode(dev)
    def circuit_qpe(n_bits):
        hqml.FockState(4, target_wires)
        estimation_wires = range(n_bits)
        QuantumPhaseEstimation(U, estimation_wires=estimation_wires)
        return hqml.expval(qml.Z(0))

    hqml.draw_mpl(circuit_qpe, style="sketch", level="device")(2)
    plt.savefig(folder / "qpe_circuit.png", dpi=300)


if __name__ == "__main__":
    ex_jc_circuit()
    colored_circuit()
    no_icons()
    banner_circuit()
    quantum_phase_estimation()
