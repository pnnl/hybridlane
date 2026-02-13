# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import warnings

import numpy as np
import pennylane as qml
import pytest
from pennylane.decomposition.symbolic_decomposition import pow_rotation
from pennylane.operation import Operation
from pennylane.templates import QuantumPhaseEstimation

import hybridlane as hqml
from hybridlane.ops.mixins import Hybrid
from hybridlane.sa import BasisSchema, ComputationalBasis


@pytest.fixture(scope="class", autouse=True)
def graph_enabled():
    qml.decomposition.enable_graph()
    yield
    qml.decomposition.disable_graph()


# Necessary to define this class so we can give the pow_rotation decomposition. Otherwise,
# pennylane uses the pow_repeat_base decomposition and that dramatically expands the circuit depth
class DJCEvo(Operation, Hybrid):
    num_wires = 2
    num_qumodes = 1
    num_params = 1

    resource_keys = set()

    def __init__(self, t: float, omega_r=1, omega_q=-1, chi=0.1, wires=None, id=None):
        self.hyperparameters.update(
            {"omega_r": omega_r, "omega_q": omega_q, "chi": chi}
        )
        super().__init__(t, wires=wires, id=id)


@qml.register_resources({qml.RZ: 1, hqml.Rotation: 1, hqml.ConditionalRotation: 1})
def _djc_decomp(t, wires, omega_r, omega_q, chi, **_):
    qml.RZ(-omega_q * t, wires[0])
    hqml.Rotation(omega_r * t, wires[1])
    hqml.ConditionalRotation(-chi * t, wires)


qml.add_decomps(DJCEvo, _djc_decomp)
qml.add_decomps("Pow(DJCEvo)", pow_rotation)


class TestApplications:
    def test_dispersive_jc_qpe(self):
        omega_r = 1
        omega_q = -1
        chi = 0.1
        U = DJCEvo(1, omega_r, omega_q, chi, ("q", "m"))

        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @qml.transforms.decompose(
            gate_set={
                hqml.Red,
                hqml.Blue,
                hqml.ConditionalRotation,
                hqml.Rotation,
                qml.RZ,
                qml.CRZ,
                qml.CNOT,
                qml.H,
                qml.ControlledPhaseShift,
            },
        )
        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit(n_bits: int):
            hqml.FockState(4, wires=("q", "m"))
            estimation_wires = range(n_bits)
            QuantumPhaseEstimation(U, estimation_wires=estimation_wires)

            schema = BasisSchema({estimation_wires: ComputationalBasis.Discrete})
            return hqml.sample(schema=schema)

        # Decomposition raises a warning if it can't find a decomposition
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            specs = qml.specs(circuit, level="device")(5)

        gate_types = specs["resources"].gate_types
        assert gate_types["Hadamard"] == 10  # 2 per estimation bit
        assert gate_types["ConditionalRotation"] == 15  # 2 per CR term, 1 per R term
        assert gate_types["Rotation"] == 5  # 1 per R term


class TestGateDecompositions:
    def test_snap_to_sqr(self):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=32)

        @qml.qnode(dev)
        def circuit():
            hqml.Displacement(0.5, 0, 0)
            hqml.SNAP(0.5, 1, [1, 0])
            hqml.SNAP(0.5, 2, [1, 0])
            hqml.SNAP(0.5, 3, [1, 0])
            hqml.Displacement(-0.5, 0, 0)
            return hqml.expval(hqml.X(0))

        expval_with_snap = circuit()

        sqr_circuit = qml.transforms.decompose(
            circuit, gate_set={hqml.Displacement, hqml.SQR}
        )
        expval_with_sqr = sqr_circuit()

        assert np.allclose(expval_with_sqr, expval_with_snap)
