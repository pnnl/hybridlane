# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import math
import sys
import textwrap
from types import ModuleType
from unittest import mock

import numpy as np
import pennylane as qml
import pytest

jaqalpaq = pytest.importorskip("jaqalpaq")

from jaqalpaq.parser import parse_jaqal_string  # noqa: E402

import hybridlane as hqml  # noqa: E402
from hybridlane.devices.sandia_qscout import to_jaqal  # noqa: E402
from hybridlane.devices.sandia_qscout.jaqal import (  # noqa: E402
    QUBIT_BOSON_MODULE,
    get_boson_gate_defs,
)


@pytest.fixture(scope="class", autouse=True)
def graph_enabled():
    qml.decomposition.enable_graph()
    yield
    qml.decomposition.disable_graph()


def programs_equal(actual_ir, expected_ir):
    # Fake having the qubit boson gate definitions
    gates = get_boson_gate_defs()
    qb_module = ModuleType(QUBIT_BOSON_MODULE)
    jaqal_gates_obj = mock.Mock()
    jaqal_gates_obj.ALL_GATES = gates
    qb_module.jaqal_gates = jaqal_gates_obj
    calibration_module_name = QUBIT_BOSON_MODULE.split(".")[0]
    calibration_module = ModuleType(calibration_module_name)
    calibration_module.QubitBosonPulses = qb_module
    with mock.patch.dict(
        sys.modules,
        {
            calibration_module_name: calibration_module,
            QUBIT_BOSON_MODULE: qb_module,
        },
    ):
        actual_program = parse_jaqal_string(actual_ir)
        expected_program = parse_jaqal_string(expected_ir)
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
                from Calibration_PulseDefinitions.QubitBosonPulses usepulses *
                register q[2]
                subcircuit 20 {{
                    Rz q[{wires[0]}] 3.141592653589793
                    Ry q[{wires[0]}] 3.141592653589793
                    XX q[{wires[0]}] q[{wires[1]}] 1.5707963267948966
                    Rx q[{wires[1]}] 10.995574287564276
                    Rz q[{wires[0]}] 7.853981633974483
                    Ry q[{wires[0]}] 1.5707963267948968
                    Rz q[{wires[0]}] 1.5707963267948966
                }}
                """
            ).strip()

        assert any(
            programs_equal(actual_ir, prog)
            for prog in (get_valid_programs(0, 1), get_valid_programs(1, 0))
        )

    def test_red_blue_gates(self):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=2)

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            hqml.JC(0.5, 0, [0, "m1i1"])
            hqml.AJC(0.5, 0, [0, "m1i1"])
            return hqml.expval(qml.Z(0))

        actual_ir = to_jaqal(circuit, level="device")()
        expected_ir = textwrap.dedent(
            r"""
            from qscout.v1.std usepulses *
            from Calibration_PulseDefinitions.QubitBosonPulses usepulses *
            register q[1]
            subcircuit 20 {
               	JC q[0] 1 1 0.0 0.5
               	AJC q[0] 1 1 0.0 0.5
            }
            """
        ).strip()

        assert programs_equal(actual_ir, expected_ir)

    def test_catstate_circuit(self):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=2)

        @qml.set_shots(1024)
        @qml.qnode(dev)
        def circuit(alpha):
            hqml.SqueezedCatState(alpha, np.pi / 2, parity="even", wires=["q", "m1i1"])

        actual_ir = to_jaqal(circuit, level="device", precision=4)(4)
        expected_ir = textwrap.dedent(
            r"""
            from qscout.v1.std usepulses *
            from Calibration_PulseDefinitions.QubitBosonPulses usepulses *
            register q[2]
            subcircuit 1024 {
               	xCD q[1] 1 1 4.0 0.0
               	Rz q[1] 11.00
               	xCD q[1] 1 1 0.00000000000000001803 0.09817
               	Rz q[1] 11.00
               	xCD q[1] 1 1 -0.09817 -0.0
               	Rz q[1] 3.142
            }
            """
        )

        assert programs_equal(actual_ir, expected_ir)

    def test_dynamic_displacement_decomposition(self):
        dev = qml.device("sandiaqscout.hybrid", optimize=True, n_qubits=2)

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit(dist):
            hqml.XCD(dist, 0, [0, "m"])
            hqml.D(dist, math.pi / 2, "m")
            hqml.XCD(-dist, 0, [0, "m"])
            hqml.D(-dist, math.pi / 2, "m")
            return hqml.expval(qml.Z(0))

        actual_ir = to_jaqal(circuit, level="device", precision=4)(1.0)
        expected_ir = textwrap.dedent(
            r"""
            from qscout.v1.std usepulses *
            from Calibration_PulseDefinitions.QubitBosonPulses usepulses *
            register q[2]
            subcircuit 20 {
               	xCD q[0] 1 1 1.0 0.0
               	zCD q[1] 1 1 0.00000000000000006123 1.0
               	xCD q[0] 1 1 -1.0 -0.0
               	zCD q[1] 1 1 -0.00000000000000006123 -1.0
            }
            """
        ).strip()

        assert programs_equal(actual_ir, expected_ir)
