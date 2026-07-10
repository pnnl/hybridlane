# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

import math
import sys
from collections import Counter

import pennylane as qp
import pytest
from pennylane.exceptions import DecompositionWarning, DeviceError
from pennylane.wires import Wires
from pennylane.workflow import construct_tape

import hybridlane as hl
from hybridlane import wires as sa
from hybridlane.devices.sandia_qscout import Qumode
from hybridlane.devices.sandia_qscout import ops as ion


@pytest.mark.unit
def test_package_works_without_jaqalpaq(monkeypatch):
    monkeypatch.delitem(sys.modules, "jaqalpaq", raising=False)
    monkeypatch.delitem(sys.modules, "qscout", raising=False)

    import hybridlane  # noqa: F401


@pytest.mark.usefixtures("enable_graph_decomp")
class TestDevice:
    @pytest.mark.unit
    @pytest.mark.parametrize("allow_com", (True, False))
    def test_com_modes(self, allow_com):
        dev = qp.device(
            "sandiaqscout.hybrid", enable_com_modes=allow_com, use_virtual_wires=False
        )

        qubits = dev._max_qubits

        if allow_com:
            qumodes = 2 * qubits
        else:
            qumodes = 2 * qubits - 2
            assert len(dev.wires & [Qumode(0, 0), Qumode(1, 0)]) == 0

        assert len(dev.wires) == qubits + qumodes

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "obs",
        (
            qp.X(0) @ qp.X(1),
            qp.X(0) @ qp.Y(3) @ qp.Z(1),
            qp.Z(0),
            qp.s_prod(0.5, qp.Z(0) @ qp.X(1)),
        ),
    )
    def test_supported_sample_observable(self, obs):
        dev = qp.device("sandiaqscout.hybrid")

        @qp.set_shots(20)
        @qp.qnode(dev)
        def circuit(obs):
            return hl.var(obs)

        circuit(obs)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "obs",
        (
            qp.X(0) @ qp.X(1),
            qp.X(0) @ qp.Y(3) @ qp.Z(1),
            qp.Z(0),
            qp.s_prod(0.5, qp.Z(0) @ qp.X(1)),
        ),
    )
    def test_no_analytic_measurement(self, obs):
        dev = qp.device("sandiaqscout.hybrid")

        @qp.qnode(dev)
        def circuit(obs):
            return hl.expval(obs)

        with pytest.raises(DeviceError):
            circuit(obs)

    @pytest.mark.integration
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
        dev = qp.device("sandiaqscout.hybrid", n_qubits=6, use_virtual_wires=False)

        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit(bs_wires):
            ion.NativeBeamsplitter(1, 2, 3, 4, wires=bs_wires)
            return hl.expval(qp.Z(0))

        if allowed:
            circuit(wires)
        else:
            with pytest.raises(DeviceError):
                circuit(wires)

    @pytest.mark.integration
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
        dev = qp.device("sandiaqscout.hybrid", n_qubits=6, use_virtual_wires=False)

        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit(wires):
            ion.ConditionalXSqueezing(0.5, wires=wires)
            return hl.expval(qp.Z(0))

        if allowed:
            circuit(wires)
        else:
            with pytest.raises(DeviceError):
                circuit(wires)

    @pytest.mark.unit
    def too_many_qubits(self):
        dev = qp.device("sandiaqscout.hybrid")

        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit():
            for i in range(dev._max_qubits):
                qp.H(i)
            return hl.expval(qp.Z(0))

        with pytest.raises(DeviceError):
            circuit()

    @pytest.mark.unit
    def too_many_qumodes(self):
        dev = qp.device("sandiaqscout.hybrid")

        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit():
            for i in range(2 * dev._max_qubits - 2 + 1):
                hl.Blue(0.5, 0.5, [0, i + 1])
            return hl.expval(qp.Z(0))

        with pytest.raises(DeviceError):
            circuit()


@pytest.mark.usefixtures("enable_graph_decomp")
class TestLayout:
    @pytest.mark.integration
    def test_qumode_assignment(self):
        dev = qp.device("sandiaqscout.hybrid", n_qubits=4, use_virtual_wires=True)

        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit():
            for i in range(5):
                hl.Blue(0.5, 0, wires=[0, i + 1])
            return hl.expval(qp.Z(0))

        tape = construct_tape(circuit, level="device")()
        sa_res = sa.type_check(tape)

        allowed_wires = Wires([Qumode(m, i) for m in (0, 1) for i in range(1, 4)])
        assert allowed_wires.contains_wires(sa_res.qumodes)

    @pytest.mark.integration
    def test_qumode_assignment_with_com(self):
        dev = qp.device("sandiaqscout.hybrid", n_qubits=3, enable_com_modes=True)

        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit():
            for i in range(5):
                hl.Blue(0.5, 0, wires=[0, i + 1])
            return hl.expval(qp.Z(0))

        tape = construct_tape(circuit, level="device")()
        sa_res = sa.type_check(tape)

        allowed_wires = Wires([Qumode(m, i) for m in (0, 1) for i in range(0, 4)])
        assert allowed_wires.contains_wires(sa_res.qumodes)

    @pytest.mark.integration
    def test_no_valid_assignment(self):
        dev = qp.device("sandiaqscout.hybrid", n_qubits=3, enable_com_modes=True)

        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit():
            # Both are hardcoded to the tilt modes, but there's only 2 tilt modes, not 3
            ion.NativeBeamsplitter(0.1, 0.2, 0.3, 0.4, [0, "m1", "m2"])
            ion.ConditionalXSqueezing(0.5, [1, "m3"])

        with pytest.raises(DeviceError):
            construct_tape(circuit, level="device")()


@pytest.mark.usefixtures("enable_graph_decomp")
@pytest.mark.integration
class TestDecomposition:
    def test_fockstate_and_conditionaldisplacement(self):
        dev = qp.device(
            "sandiaqscout.hybrid", optimize=True, use_virtual_wires=False, n_qubits=3
        )

        @qp.set_shots(20)
        @qp.qnode(dev)
        def circuit():
            hl.FockState(5, [0, "m0i1"])
            hl.FockState(5, [0, "m0i2"])
            hl.ConditionalDisplacement(0.5, 0.5, [0, "m1i1"])
            hl.ConditionalDisplacement(-0.5, -0.5, [0, "m1i1"])
            return hl.expval(qp.X(0))

        tape = construct_tape(circuit, level="device")()
        op_counts = Counter([type(op) for op in tape.operations])

        assert op_counts[hl.FockState] == 2

        # Check the conditional displacements are optimized away
        assert op_counts[hl.CD] == 0

    def test_cnot_to_xx(self):
        dev = qp.device("sandiaqscout.hybrid")

        @qp.set_shots(20)
        @qp.qnode(dev)
        def circuit():
            qp.H(0)
            qp.CNOT([0, 1])
            return hl.expval(qp.X(0))

        tape = construct_tape(circuit, level="device")()
        op_counts = Counter([type(op) for op in tape.operations])
        assert op_counts[qp.IsingXX] == 1

    def test_no_beamsplitter_decomposition(self):
        dev = qp.device(
            "sandiaqscout.hybrid", n_qubits=6, optimize=True, use_virtual_wires=False
        )

        @qp.set_shots(20)
        @qp.qnode(dev)
        def circuit():
            # Hybridlane Beamsplitter gate isn't defined, instead one has to use
            # NativeBeamsplitter
            hl.ModeSwap(wires=["m1i1", "m0i3"])
            return hl.expval(qp.Z(0))

        # Emits 2 warnings if it can't find a decomposition, then raises an error
        # because the decomposed tape has non-native operations
        with (
            pytest.warns(
                DecompositionWarning, match="unable to find a decomposition for"
            ),
            pytest.raises(DeviceError),
        ):
            construct_tape(circuit, level="device")()

    def test_no_squeezing_decomposition(self):
        dev = qp.device(
            "sandiaqscout.hybrid", n_qubits=6, optimize=True, use_virtual_wires=False
        )

        @qp.set_shots(20)
        @qp.qnode(dev)
        def circuit():
            # Hybridlane CS gate isn't defined, instead one has to use
            # ConditionalXSqueezing
            hl.ConditionalSqueezing(1, 0, wires=[0, "m1i1"])
            return hl.expval(qp.Z(0))

        # Emits 2 warnings if it can't find a decomposition, then raises an error
        # because the decomposed tape has non-native operations
        with (
            pytest.warns(
                DecompositionWarning, match="unable to find a decomposition for"
            ),
            pytest.raises(DeviceError),
        ):
            construct_tape(circuit, level="device")()

    def test_dynamic_displacement_decomposition(self):
        dev = qp.device("sandiaqscout.hybrid", optimize=True, n_qubits=6)

        @qp.set_shots(20)
        @qp.qnode(dev)
        def circuit(dist):
            qp.H("q")
            hl.CD(dist, 0, ["q", "m"])
            hl.D(dist, math.pi / 2, "m")
            hl.CD(-dist, 0, ["q", "m"])
            hl.D(-dist, math.pi / 2, "m")
            qp.H("q")
            return hl.expval(qp.Z("q"))

        specs = qp.specs(circuit, level="device")(1.0)
        gate_count = specs.resources.gate_types  # ty:ignore[unresolved-attribute]

        # The D gates should be replaced by CD gates on an ancilla qubit
        assert gate_count.get("Displacement", 0) == 0
        assert gate_count["ConditionalXDisplacement"] == 4
