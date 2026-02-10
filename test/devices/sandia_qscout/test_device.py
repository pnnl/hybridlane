# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import importlib
import sys
from collections import Counter

import pennylane as qml
import pytest
from pennylane.exceptions import DeviceError
from pennylane.wires import Wires
from pennylane.workflow import construct_tape

import hybridlane as hqml
from hybridlane import sa
from hybridlane.devices.sandia_qscout import Qumode
from hybridlane.devices.sandia_qscout import ops as ion


@pytest.fixture(scope="class", autouse=True)
def graph_enabled():
    qml.decomposition.enable_graph()
    yield
    qml.decomposition.disable_graph()


def test_package_works_without_jaqalpaq(monkeypatch):
    monkeypatch.delitem(sys.modules, "jaqalpaq", raising=False)
    monkeypatch.delitem(sys.modules, "qscout", raising=False)

    import hybridlane  # noqa: F401


missing_jaqalpaq = importlib.util.find_spec("jaqalpaq") is None


@pytest.mark.skipif(missing_jaqalpaq, reason="jaqalpaq is not installed")
class TestDevice:
    @pytest.mark.parametrize("allow_com", (True, False))
    def test_com_modes(self, allow_com):
        dev = qml.device(
            "sandiaqscout.hybrid", enable_com_modes=allow_com, use_virtual_wires=False
        )

        qubits = dev._max_qubits

        if allow_com:
            qumodes = 2 * qubits
        else:
            qumodes = 2 * qubits - 2
            assert len(dev.wires & [Qumode(0, 0), Qumode(1, 0)]) == 0

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
        dev = qml.device("sandiaqscout.hybrid")

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
        dev = qml.device("sandiaqscout.hybrid")

        @qml.qnode(dev)
        def circuit(obs):
            return hqml.expval(obs)

        with pytest.raises(DeviceError):
            circuit(obs)

    @pytest.mark.parametrize(
        "wires, allowed",
        (
            ([0, "m1i1", "m1i2"], False),
            ([0, "m1i3", "m1i1"], False),
            ([0, "m1i1", "m0i1"], True),
            ([1, "m1i1", "m0i1"], True),
            ([2, "m1i1", "m0i1"], False),
        ),
    )
    def test_beamsplitter_constraints(self, wires, allowed):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=6, use_virtual_wires=False)

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
            ([0, "m1i1"], True),
            ([1, "m1i1"], True),
            ([2, "m1i1"], True),
            ([0, "m1i3"], False),
            ([0, "m0i2"], False),
            ([0, "m0i3"], False),
        ),
    )
    def test_rampup_constraints(self, wires, allowed):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=6, use_virtual_wires=False)

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
        dev = qml.device("sandiaqscout.hybrid")

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            for i in range(dev._max_qubits):
                qml.H(i)
            return hqml.expval(qml.Z(0))

        with pytest.raises(DeviceError):
            circuit()

    def too_many_qumodes(self):
        dev = qml.device("sandiaqscout.hybrid")

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            for i in range(2 * dev._max_qubits - 2 + 1):
                hqml.Blue(0.5, 0.5, [0, i + 1])
            return hqml.expval(qml.Z(0))

        with pytest.raises(DeviceError):
            circuit()


@pytest.mark.skipif(missing_jaqalpaq, reason="jaqalpaq is not installed")
class TestLayout:
    def test_qumode_assignment(self):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=4, use_virtual_wires=True)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            for i in range(5):
                hqml.Blue(0.5, 0, wires=[0, i + 1])
            return hqml.expval(qml.Z(0))

        tape = construct_tape(circuit, level="device")()
        sa_res = sa.analyze(tape)

        allowed_wires = Wires([Qumode(m, i) for m in (0, 1) for i in range(1, 4)])
        assert allowed_wires.contains_wires(sa_res.qumodes)

    def test_qumode_assignment_with_com(self):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=3, enable_com_modes=True)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            for i in range(5):
                hqml.Blue(0.5, 0, wires=[0, i + 1])
            return hqml.expval(qml.Z(0))

        tape = construct_tape(circuit, level="device")()
        sa_res = sa.analyze(tape)

        allowed_wires = Wires([Qumode(m, i) for m in (0, 1) for i in range(0, 4)])
        assert allowed_wires.contains_wires(sa_res.qumodes)

    def test_no_valid_assignment(self):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=3, enable_com_modes=True)

        @qml.set_shots(10)
        @qml.qnode(dev)
        def circuit():
            # Both are hardcoded to the tilt modes, but there's only 2 tilt modes, not 3
            ion.NativeBeamsplitter(0.1, 0.2, 0.3, 0.4, [0, "m1", "m2"])
            ion.ConditionalXSqueezing(0.5, [1, "m3"])

        with pytest.raises(DeviceError):
            construct_tape(circuit, level="device")()


@pytest.mark.skipif(missing_jaqalpaq, reason="jaqalpaq is not installed")
class TestDecomposition:
    def test_fockladder_and_conditionaldisplacement(self):
        dev = qml.device(
            "sandiaqscout.hybrid", optimize=True, use_virtual_wires=False, n_qubits=3
        )

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            hqml.FockLadder(5, [0, "m0i1"])
            hqml.FockLadder(5, [0, "m0i2"])
            hqml.ConditionalDisplacement(0.5, 0.5, [0, "m1i1"])
            hqml.ConditionalDisplacement(-0.5, -0.5, [0, "m1i1"])
            return hqml.expval(qml.X(0))

        tape = construct_tape(circuit, level="device")()
        op_counts = Counter([type(op) for op in tape.operations])

        # Both fock ladders get converted into native instruction
        assert op_counts[ion.FockStatePrep] == 2
        assert isinstance(tape.operations[0], ion.FockStatePrep)

        # Check the conditional displacements are optimized away
        assert op_counts[hqml.CD] == 0

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
            # Hybridlane Beamsplitter gate isn't defined, instead one has to use
            # NativeBeamsplitter
            hqml.ModeSwap(wires=["m1i1", "m0i3"])
            return hqml.expval(qml.Z(0))

        # Emits 2 warnings if it can't find a decomposition
        with pytest.warns(UserWarning):
            with pytest.warns(UserWarning, match="unable to find a decomposition for"):
                construct_tape(circuit, level="device")()

    def test_no_squeezing_decomposition(self):
        dev = qml.device("sandiaqscout.hybrid", n_qubits=6, optimize=True)

        @qml.set_shots(20)
        @qml.qnode(dev)
        def circuit():
            # Hybridlane CS gate isn't defined, instead one has to use ConditionalXSqueezing
            hqml.ConditionalSqueezing(1, 0, wires=[0, "m1i1"])
            return hqml.expval(qml.Z(0))

        # Emits 2 warnings if it can't find a decomposition
        with pytest.warns(UserWarning):
            with pytest.warns(UserWarning, match="unable to find a decomposition for"):
                construct_tape(circuit, level="device")()
