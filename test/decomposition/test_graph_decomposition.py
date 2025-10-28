import warnings
from functools import partial

import pennylane as qml
import pytest
from pennylane.decomposition.symbolic_decomposition import pow_rotation
from pennylane.operation import Operation
from pennylane.templates import QuantumPhaseEstimation
from pennylane.wires import Wires

import hybridlane as hqml
from hybridlane.ops.mixins import Hybrid
from hybridlane.sa import BasisSchema, ComputationalBasis


@pytest.fixture(scope="class")
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

    @property
    def resource_params(self):
        return {}


@qml.register_resources({qml.RZ: 1, hqml.Rotation: 1, hqml.ConditionalRotation: 1})
def _djc_decomp(t, wires, omega_r, omega_q, chi, **_):
    qml.RZ(-omega_q * t, wires[0])
    hqml.Rotation(omega_r * t, wires[1])
    hqml.ConditionalRotation(-chi * t, wires)


qml.add_decomps(DJCEvo, _djc_decomp)
qml.add_decomps("Pow(DJCEvo)", pow_rotation)


class TestApplications:
    def test_dispersive_jc_qpe(self, graph_enabled):
        omega_r = 1
        omega_q = -1
        chi = 0.1
        U = DJCEvo(1, omega_r, omega_q, chi, ("q", "m"))

        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @partial(
            qml.transforms.decompose,
            gate_set={
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
            estimation_wires = range(n_bits)
            QuantumPhaseEstimation(U, estimation_wires=estimation_wires)

            schema = BasisSchema(
                {Wires(w): ComputationalBasis.Discrete for w in estimation_wires}
            )
            return hqml.sample(schema=schema)

        # Decomposition raises a warning if it can't find a decomposition
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            specs = qml.specs(circuit, level="user")(5)

        gate_types = specs["resources"].gate_types
        assert gate_types["Hadamard"] == 10  # 2 per estimation bit
        assert gate_types["ConditionalRotation"] == 15  # 2 per CR term, 1 per R term
        assert gate_types["Rotation"] == 5  # 1 per R term
