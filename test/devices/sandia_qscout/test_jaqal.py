# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import textwrap

import pennylane as qml
import pytest

import hybridlane as hqml
from hybridlane.devices.sandia_qscout import QscoutIonTrap, to_jaqal


@pytest.fixture(scope="class", autouse=True)
def graph_enabled():
    qml.decomposition.enable_graph()
    yield
    qml.decomposition.disable_graph()


class TestToJaqal:
    def test_sample_qubit_circuit(self):
        dev = QscoutIonTrap()

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            qml.CNOT([0, 1])
            return hqml.expval(qml.X(0))

        program = to_jaqal(circuit, level="device")()

        def get_valid_programs(*wires):
            return textwrap.dedent(
                f"""
            from qscout.v1.std usepulses *

            register q[2]

            subcircuit 20 {{
            \tRz q[{wires[0]}] 3.141592653589793
            \tRy q[{wires[0]}] 3.141592653589793
            \tXX q[{wires[0]}] q[{wires[1]}] 1.5707963267948966
            \tRx q[{wires[1]}] 10.995574287564276
            \tRz q[{wires[0]}] 7.853981633974483
            \tRy q[{wires[0]}] 1.5707963267948968
            \tRz q[{wires[0]}] 1.5707963267948966
            }}
            """
            ).strip()

        assert program in {get_valid_programs(0, 1), get_valid_programs(1, 0)}

    def test_catstate_circuit(self):
        dev = QscoutIonTrap(n_qubits=2)

        @qml.set_shots(1024)
        @qml.qnode(dev)
        def circuit(alpha):
            hqml.SqueezedCatState(alpha, 0, parity="even", wires=["q", "m"])

        to_jaqal(circuit, level="device", precision=4)(4)
