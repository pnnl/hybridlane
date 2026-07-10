# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

import math
import sys
import textwrap
from types import ModuleType
from unittest import mock

import numpy as np
import pennylane as qp
import pytest

jaqalpaq = pytest.importorskip("jaqalpaq")

from jaqalpaq.parser import parse_jaqal_string  # noqa: E402

import hybridlane as hl  # noqa: E402
from hybridlane.devices.sandia_qscout import to_jaqal  # noqa: E402
from hybridlane.devices.sandia_qscout.jaqal import (  # noqa: E402
    QUBIT_BOSON_MODULE,
    get_boson_gate_defs,
)


def programs_equal(actual_ir, expected_ir):
    # Fake the qubit pulses as these are only valid on the emulator, not hardware
    import_statement = "from qscout.v1.std usepulses *\n"
    actual_ir = import_statement + actual_ir
    expected_ir = import_statement + expected_ir

    # Fake having the qubit boson gate definitions
    gates = get_boson_gate_defs()
    qb_module = ModuleType(QUBIT_BOSON_MODULE)
    jaqal_gates_obj = mock.Mock()
    jaqal_gates_obj.ALL_GATES = gates
    qb_module.jaqal_gates = jaqal_gates_obj  # ty:ignore[unresolved-attribute]
    calibration_module_name = QUBIT_BOSON_MODULE.split(".")[0]
    calibration_module = ModuleType(calibration_module_name)
    calibration_module.QubitBosonPulses = qb_module  # ty:ignore[unresolved-attribute]
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


@pytest.mark.integration
@pytest.mark.usefixtures("enable_graph_decomp")
class TestToJaqal:
    def test_sample_qubit_circuit(self):
        dev = qp.device("sandiaqscout.hybrid")

        @qp.set_shots(20)
        @qp.qnode(dev)
        def circuit():
            qp.H(0)
            qp.CNOT([0, 1])
            return hl.expval(qp.X(0))

        actual_ir = to_jaqal(circuit, level="device", precision=4)()
        expected_ir = textwrap.dedent(
            r"""
            from Calibration_PulseDefinitions.QubitBosonPulses usepulses *

            register q[2]

            subcircuit {
                Rz q[0] 3.142
                Ry q[0] 3.142
                XX q[0] q[1] 1.571
                Rx q[1] 11.00
                Rz q[0] 7.854
                Ry q[0] 1.571
                Rz q[0] 1.571
            }
            """
        ).strip()

        assert programs_equal(actual_ir, expected_ir)

    def test_red_blue_gates(self):
        dev = qp.device("sandiaqscout.hybrid", n_qubits=2)

        @qp.set_shots(20)
        @qp.qnode(dev)
        def circuit():
            hl.JC(0.5, 0, [0, "m1i1"])
            hl.AJC(0.5, 0, [0, "m1i1"])
            return hl.expval(qp.Z(0))

        actual_ir = to_jaqal(circuit, level="device")()
        expected_ir = textwrap.dedent(
            r"""
            from Calibration_PulseDefinitions.QubitBosonPulses usepulses *

            register q[1]

            subcircuit {
                JC q[0] 1 1 0.0 0.5
                AJC q[0] 1 1 0.0 0.5
            }
            """
        ).strip()

        assert programs_equal(actual_ir, expected_ir)

    def test_catstate_circuit(self):
        dev = qp.device("sandiaqscout.hybrid", n_qubits=2)

        @qp.set_shots(1024)
        @qp.qnode(dev)
        def circuit(alpha):
            hl.SqueezedCatState(alpha, np.pi / 2, parity="even", wires=["q", "m1i1"])

        actual_ir = to_jaqal(circuit, level="device", precision=4)(4)
        expected_ir = textwrap.dedent(
            r"""
            from Calibration_PulseDefinitions.QubitBosonPulses usepulses *

            register q[2]

            subcircuit {
                xCD q[1] 1 1 4.0 0.0
                Rz q[1] 11.00
                xCD q[1] 1 1 0.0 0.09818
                yCD q[1] 1 1 -0.09818 -0.0
                Sz q[1]
            }
            """
        )

        assert programs_equal(actual_ir, expected_ir)

    def test_dynamic_displacement_decomposition(self):
        dev = qp.device("sandiaqscout.hybrid", optimize=False, n_qubits=2)

        @qp.set_shots(20)
        @qp.qnode(dev)
        def circuit(dist):
            hl.XCD(dist, 0, [0, "m"])
            hl.D(dist, math.pi / 2, "m")
            hl.XCD(-dist, 0, [0, "m"])
            hl.D(-dist, math.pi / 2, "m")
            return hl.expval(qp.Z(0))

        actual_ir = to_jaqal(circuit, level="device", precision=4)(1.0)
        expected_ir = textwrap.dedent(
            r"""
            from Calibration_PulseDefinitions.QubitBosonPulses usepulses *
            register q[2]
            subcircuit {
                xCD q[0] 1 1 1.0 0.0
                zCD q[1] 1 1 0.0 1.0
                xCD q[0] 1 1 -1.0 -0.0
                zCD q[1] 1 1 -0.0 -1.0
            }
            """
        ).strip()

        assert programs_equal(actual_ir, expected_ir)
