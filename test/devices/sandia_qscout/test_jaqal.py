# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import textwrap

import pennylane as qml
import pytest

jaqalpaq = pytest.importorskip("jaqalpaq")

from jaqalpaq.parser import parse_jaqal_string  # noqa: E402

import hybridlane as hqml  # noqa: E402
from hybridlane.devices.sandia_qscout import to_jaqal  # noqa: E402
from hybridlane.devices.sandia_qscout.jaqal import get_boson_gate_defs  # noqa: E402


@pytest.fixture(scope="class", autouse=True)
def graph_enabled():
    qml.decomposition.enable_graph()
    yield
    qml.decomposition.disable_graph()


def programs_equal(actual_ir, expected_ir):
    gates = get_boson_gate_defs()
    actual_program = parse_jaqal_string(actual_ir, inject_pulses=gates)
    expected_program = parse_jaqal_string(expected_ir, inject_pulses=gates)
    return actual_program == expected_program


class TestToJaqal:
    def test_sample_qubit_circuit(self):
        dev = qml.device("sandiaqscout.hybrid")

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            qml.CNOT([0, 1])
            return hqml.expval(qml.X(0))

        actual_ir = to_jaqal(circuit, level="device")()

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

        assert any(
            programs_equal(actual_ir, prog)
            for prog in (get_valid_programs(0, 1), get_valid_programs(1, 0))
        )

    def test_catstate_circuit(self):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=2)

        @qml.set_shots(1024)
        @qml.qnode(dev)
        def circuit(alpha):
            hqml.SqueezedCatState(alpha, 0, parity="even", wires=["q", "m1i1"])

        actual_ir = to_jaqal(circuit, level="device", precision=4)(4)
        expected_ir = textwrap.dedent(
            r"""
            from qscout.v1.std usepulses *

            register q[2]

            subcircuit 1024 {
               	xCD q[1] 1 1 4.0 0.0
               	Rx q[1] 9.425
               	R q[1] 2.356 3.142
               	xCD q[1] 1 1 0.00000000000000001803 0.09817
               	Rx q[1] 9.425
               	R q[1] 2.356 3.142
               	xCD q[1] 1 1 -0.09817 -0.0
               	Ry q[1] 9.425
               	Rx q[1] 3.142
            }
            """
        )

        assert programs_equal(actual_ir, expected_ir)
