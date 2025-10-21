# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

from collections import Counter
from functools import partial

import pennylane as qml
import pytest
from pennylane.decomposition import DecompositionError
from pennylane.exceptions import DeviceError
from pennylane.wires import WireError, Wires
from pennylane.workflow import construct_tape

import hybridlane as hqml
from hybridlane import sa
from hybridlane.devices.sandia_qscout import QscoutIonTrap
from hybridlane.devices.sandia_qscout import ops as ion


@pytest.fixture(scope="class", autouse=True)
def graph_enabled():
    qml.decomposition.enable_graph()
    yield
    qml.decomposition.disable_graph()


class TestDevice:
    @pytest.mark.parametrize("allow_com", (True, False))
    def test_com_modes(self, allow_com):
        dev = QscoutIonTrap(use_com_modes=allow_com, use_hardware_wires=True)

        qubits = dev._max_qubits

        if allow_com:
            qumodes = 2 * qubits
        else:
            qumodes = 2 * qubits - 2
            assert len(dev.wires & ["a0m0", "a1m0"]) == 0

        assert len(dev.wires) == qubits + qumodes

    @pytest.mark.parametrize(
        "obs",
        (
            qml.X(0) @ qml.X(1),
            qml.X(0) @ qml.Y(3) @ qml.Z(1),
            qml.Z(0),
            qml.s_prod(0.5, qml.Z(0) @ qml.X(1)),
        ),
    )
    def test_supported_sample_observable(self, obs):
        dev = QscoutIonTrap()

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit(obs):
            return hqml.var(obs)

        circuit(obs)

    @pytest.mark.parametrize(
        "obs",
        (
            qml.X(0) @ qml.X(1),
            qml.X(0) @ qml.Y(3) @ qml.Z(1),
            qml.Z(0),
            qml.s_prod(0.5, qml.Z(0) @ qml.X(1)),
        ),
    )
    def test_no_analytic_measurement(self, obs):
        dev = QscoutIonTrap()

        @qml.qnode(dev)
        def circuit(obs):
            return hqml.expval(obs)

        with pytest.raises(DeviceError):
            circuit(obs)

    @pytest.mark.parametrize(
        "wires, allowed",
        (
            ([0, "a0m1", "a0m2"], False),
            ([0, "a0m3", "a0m1"], False),
            ([0, "a0m1", "a1m1"], True),
            ([1, "a0m1", "a1m1"], True),
            ([2, "a0m1", "a1m1"], True),
        ),
    )
    def test_beamsplitter_constraints(self, wires, allowed):
        dev = QscoutIonTrap(n_qubits=6, use_hardware_wires=True)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit(bs_wires):
            ion.NativeBeamsplitter(1, 2, 3, 4, wires=bs_wires)
            return hqml.expval(qml.Z(0))

        if allowed:
            circuit(wires)
        else:
            with pytest.raises(DeviceError):
                circuit(wires)

    @pytest.mark.parametrize(
        "wires, allowed",
        (
            ([0, "a0m1"], True),
            ([1, "a0m1"], True),
            ([2, "a0m1"], True),
            ([0, "a0m3"], False),
            ([0, "a1m2"], False),
            ([0, "a1m1"], False),
        ),
    )
    def test_rampup_constraints(self, wires, allowed):
        dev = QscoutIonTrap(n_qubits=6, use_hardware_wires=True)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit(wires):
            ion.ConditionalXSqueezing(0.5, wires=wires)
            return hqml.expval(qml.Z(0))

        if allowed:
            circuit(wires)
        else:
            with pytest.raises(DeviceError):
                circuit(wires)

    def too_many_qubits(self):
        dev = QscoutIonTrap()

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            for i in range(dev._max_qubits):
                qml.H(i)
            return hqml.expval(qml.Z(0))

        with pytest.raises(DeviceError):
            circuit()

    def too_many_qumodes(self):
        dev = QscoutIonTrap()

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            for i in range(2 * dev._max_qubits - 2 + 1):
                hqml.Blue(0.5, 0.5, [0, i + 1])
            return hqml.expval(qml.Z(0))

        with pytest.raises(DeviceError):
            circuit()


class TestLayout:
    def test_qumode_assignment(self):
        dev = QscoutIonTrap(n_qubits=4)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            for i in range(5):
                hqml.Blue(0.5, 0, wires=[0, i + 1])
            return hqml.expval(qml.Z(0))

        tape = construct_tape(circuit, level="device")()
        sa_res = sa.analyze(tape)

        assert Wires(["a0m1", "a1m1", "a0m2", "a1m2", "a0m3", "a1m3"]).contains_wires(
            sa_res.qumodes
        )

    def test_qumode_assignment_with_com(self):
        dev = QscoutIonTrap(n_qubits=3, use_com_modes=True)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            for i in range(5):
                hqml.Blue(0.5, 0, wires=[0, i + 1])
            return hqml.expval(qml.Z(0))

        tape = construct_tape(circuit, level="device")()
        sa_res = sa.analyze(tape)

        assert Wires(["a0m0", "a1m0", "a0m1", "a1m1", "a0m2", "a1m2"]).contains_wires(
            sa_res.qumodes
        )

    def test_no_valid_assignment(self):
        dev = QscoutIonTrap(n_qubits=3, use_com_modes=True)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            # Both are hardcoded to the tilt modes, but there's only 2 tilt modes, not 3
            ion.NativeBeamsplitter(0.1, 0.2, 0.3, 0.4, [0, "m1", "m2"])
            hqml.ConditionalDisplacement(0.5, 0, [1, "m3"])

        with pytest.raises(DeviceError):
            construct_tape(circuit, level="device")()

    def test_conditional_displacement(self):
        dev = QscoutIonTrap(n_qubits=3, use_com_modes=True)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            hqml.ConditionalDisplacement(0.5, 0, [1, "m1"])

        tape = construct_tape(circuit, level="device")()

        specs = qml.specs(circuit, level="device")()
        assert specs["resources"].gate_types["ConditionalXDisplacement"] == 1
        for op in tape.operations:
            if isinstance(op, ion.ConditionalXDisplacement):
                assert op.wires[1] == "a0m1"


class TestDecomposition:
    def test_fockladder_and_conditionaldisplacement(self):
        dev = qml.device(
            "sandiaqscout.hybrid", optimize=True, use_hardware_wires=True, n_qubits=3
        )

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            hqml.FockLadder(5, [0, "a1m1"])
            hqml.FockLadder(5, [0, "a1m2"])
            hqml.ConditionalDisplacement(0.5, 0.5, [0, "a0m1"])
            hqml.ConditionalDisplacement(-0.5, -0.5, [0, "a0m1"])
            return hqml.expval(qml.X(0))

        tape = construct_tape(circuit, level="device")()
        op_counts = Counter([type(op) for op in tape.operations])

        # First FockLadder has a native instruction, so it should be left alone
        assert op_counts[ion.FockStatePrep] == 1
        assert isinstance(tape.operations[0], ion.FockStatePrep)

        # Second one is non-native and should be turned into red/blue sideband pulses
        op_types = {type(op) for op in tape.operations[1:6]}
        assert op_types == {hqml.Red, hqml.Blue}

        # Check the conditional displacements got turned into SDF instructions
        assert op_counts[ion.ConditionalXDisplacement] == 2

    def test_cnot_to_xx(self):
        dev = qml.device("sandiaqscout.hybrid")

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            qml.H(0)
            qml.CNOT([0, 1])
            return hqml.expval(qml.X(0))

        tape = construct_tape(circuit, level="device")()
        op_counts = Counter([type(op) for op in tape.operations])
        assert op_counts[qml.IsingXX] == 1

    def test_no_beamsplitter_decomposition(self):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=6, optimize=True)

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            # Hybridlane Beamsplitter gate isn't defined, instead one has to use NativeBeamsplitter
            hqml.ModeSwap(wires=["a0m1", "a1m3"])
            return hqml.expval(qml.Z(0))

        with pytest.warns(UserWarning, match="unable to find a decomposition for"):
            construct_tape(circuit, level="device")()

    def test_no_squeezing_decomposition(self):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=6, optimize=True)

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            # Hybridlane CS gate isn't defined, instead one has to use ConditionalXSqueezing
            hqml.ConditionalSqueezing(1, 0, wires=[0, "a0m1"])
            return hqml.expval(qml.Z(0))

        with pytest.warns(UserWarning, match="unable to find a decomposition for"):
            construct_tape(circuit, level="device")()
