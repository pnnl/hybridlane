# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import math
from functools import partial
import textwrap

import pennylane as qml

import hybridlane as hqml
from hybridlane.devices.sandia_qscout import QscoutIonTrap, to_jaqal
from hybridlane.devices.sandia_qscout import ops as ion
from hybridlane.devices.sandia_qscout.jaqal import tokenize_operation

qml.decomposition.enable_graph()


class TestTokenizer:
    def test_fockstate(self):
        op = ion.FockStatePrep(5, wires=[0, "a0m1"])
        assert tokenize_operation(op) == "FockState q[0] 5 0"

    def test_beamsplitter(self):
        op = ion.NativeBeamsplitter(1.0, 2.0, 3.0, 4.0, wires=[0, "a0m1", "a1m1"])
        assert (
            tokenize_operation(op, precision=3) == "Beamsplitter q[0] 1.0 2.0 3.0 4.0"
        )

    def test_blue_sideband(self):
        op = hqml.Blue(0.5, -0.2, wires=[1, "a1m3"])
        assert tokenize_operation(op, precision=3) == "Blue q[1] 1 -0.2 0.5 1 3"

    def test_red_sideband(self):
        op = hqml.Red(0.5, -0.2, wires=[1, "a1m3"])
        assert tokenize_operation(op, precision=3) == "Red q[1] 1 -0.2 0.5 1 3"

    def test_sideband_probe(self):
        op = ion.SidebandProbe(0.1, 0.2, 0, 0.3, wires=[0, "a0m1"])
        assert (
            tokenize_operation(op, precision=3) == "Rt_SBProbe q[0] 0.2 0.1 0 1 0 0.3"
        )

    def test_spin_dependent_force(self):
        op = ion.ConditionalXDisplacement(0.1, -0.23, wires=[3, "a0m1"])
        assert tokenize_operation(op, precision=3) == "SDF q[3] 0.1 -0.23"

    def test_rampup(self):
        op = ion.ConditionalXSqueezing(0.2, wires=[0, "a0m1"])
        assert tokenize_operation(op, precision=3) == "RampUp q[0] 0.2"

    def test_sdg(self):
        op = qml.adjoint(qml.S(0))
        assert tokenize_operation(op) == "Szd q[0]"

    def test_sxdg(self):
        op = qml.adjoint(qml.SX(0))
        assert tokenize_operation(op) == "Sxd q[0]"

    def test_xx(self):
        op = qml.IsingXX(0.5, wires=[0, 1])
        assert tokenize_operation(op, precision=3) == "XX q[0] q[1] 0.5"

    def test_r(self):
        op = ion.R(0.5, math.pi, wires=0)
        assert tokenize_operation(op, precision=3) == "R q[0] 3.14 0.5"


class TestToJaqal:
    def test_sample_qubit_circuit(self):
        dev = QscoutIonTrap()

        @partial(qml.set_shots, shots=20)
        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            qml.CNOT([0, 1])
            return hqml.expval(qml.X(0))

        program = to_jaqal(circuit, level="device", precision=3)()

        def get_valid_programs(*wires):
            return textwrap.dedent(
                f"""
            register q[2]

            prepare_all
            Rz q[{wires[0]}] 3.14
            Ry q[{wires[0]}] 3.14
            XX q[{wires[0]}] q[{wires[1]}] 1.57
            Rx q[{wires[1]}] 11.0
            Rz q[{wires[0]}] 7.85
            Ry q[{wires[0]}] 1.57
            Rz q[{wires[0]}] 1.57

            measure_all"""
            ).strip()

        assert program in {get_valid_programs(0, 1), get_valid_programs(1, 0)}
