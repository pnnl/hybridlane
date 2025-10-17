from functools import partial

import numpy as np
import pennylane as qml
import pytest

import hybridlane as hqml
from hybridlane.ops import QubitConditioned


@pytest.fixture(scope="class")
def graph_enabled():
    qml.decomposition.enable_graph()
    yield
    qml.decomposition.disable_graph()


class TestQubitConditioned:
    def test_name(self):
        op = QubitConditioned(hqml.Rotation(0.5, 0), 1)
        assert op.name == "qCond(Rotation)"

    def test_overlapping_wires(self):
        with pytest.raises(ValueError):
            hqml.qcond(qml.RZ(0.5, 0), 0)

    def test_known_gates(self):
        # Hybridlane gates
        assert hqml.qcond(hqml.Rotation(0.5, 0), 1) == hqml.ConditionalRotation(
            1.0, [1, 0]
        )
        assert hqml.qcond(hqml.Fourier(0), 1) == hqml.ConditionalParity([1, 0])
        assert hqml.qcond(
            hqml.Beamsplitter(0.1, 0.2, [0, 1]), 2
        ) == hqml.ConditionalBeamsplitter(0.1, 0.2, [2, 0, 1])
        assert hqml.qcond(
            hqml.Displacement(0.1, 0.2, 0), 1
        ) == hqml.ConditionalDisplacement(0.1, 0.2, [1, 0])
        assert hqml.qcond(
            hqml.TwoModeSqueezing(0.1, 0, [0, 1]), 2
        ) == hqml.ConditionalTwoModeSqueezing(0.1, 0, [2, 0, 1])

        # Pennylane gates
        assert hqml.qcond(qml.GlobalPhase(0.5, 0), 1) == qml.RZ(1.0, 1)
        assert hqml.qcond(qml.GlobalPhase(0.5, 0), [1, 2]) == qml.IsingZZ(1.0, [1, 2])
        assert hqml.qcond(qml.IsingZZ(0.5, [0, 1]), [2, 3]) == qml.MultiRZ(
            0.5, [2, 3, 0, 1]
        )

    def test_nested_qcond(self):
        op = hqml.qcond(QubitConditioned(hqml.Displacement(0.5, 0, 0), 1), 2)
        assert op == QubitConditioned(hqml.Displacement(0.5, 0, 0), [2, 1])
        assert op.has_decomposition
        assert op.decomposition() == [
            qml.CNOT([2, 1]),
            hqml.ConditionalDisplacement(0.5, 0, [1, 0]),
            qml.CNOT([2, 1]),
        ]


class TestDecomposition:
    def test_rz_to_isingzz(self):
        op = QubitConditioned(qml.RZ(0.5, 0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [qml.IsingZZ(0.5, [1, 0])]

    def test_d_to_cd(self):
        op = QubitConditioned(hqml.Displacement(0.1, 0.2, 0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [hqml.ConditionalDisplacement(0.1, 0.2, [1, 0])]

    def test_f_to_cp(self):
        op = QubitConditioned(hqml.Fourier(0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [hqml.ConditionalParity([1, 0])]

    def test_r_to_cr(self):
        op = QubitConditioned(hqml.Rotation(0.5, 0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [hqml.ConditionalRotation(1.0, [1, 0])]

    def test_multirz(self):
        op = QubitConditioned(qml.MultiRZ(0.5, [0, 1]), [2, 3])
        assert op.has_decomposition
        assert op.decomposition() == [qml.MultiRZ(0.5, [2, 3, 0, 1])]

    def test_identity(self):
        op = QubitConditioned(qml.Identity(0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [qml.Identity([1, 0])]

    def test_cnot_decomposition(self):
        op = QubitConditioned(hqml.Displacement(0.1, 0.2, 0), [1, 2])
        assert op.has_decomposition
        assert op.decomposition() == [
            qml.CNOT([1, 2]),
            hqml.qcond(hqml.Displacement(0.1, 0.2, 0), 2),
            qml.CNOT([1, 2]),
        ]


class TestGraphDecomposition:
    def test_qcondf_to_cr(self, graph_enabled):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @partial(
            qml.transforms.decompose,
            gate_set={hqml.ConditionalRotation},
        )
        @qml.qnode(dev)
        def circuit():
            hqml.qcond(hqml.Fourier(0), 1)

        tape = qml.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 1  # 1 cr

    def test_multi_qcondf_to_cr(self, graph_enabled):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @partial(
            qml.transforms.decompose,
            gate_set={hqml.ConditionalRotation, qml.CNOT},
        )
        @qml.qnode(dev)
        def circuit():
            hqml.qcond(hqml.Fourier(0), [1, 2])

        tape = qml.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 3  # 1 cr, 2 cnot

    def test_ctrlf_to_cr(self, graph_enabled):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @partial(
            qml.transforms.decompose,
            gate_set={hqml.ConditionalRotation, hqml.Rotation, qml.CNOT},
        )
        @qml.qnode(dev)
        def circuit():
            qml.ctrl(hqml.Fourier(0), 1)

        tape = qml.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 2  # 1 r, 1 cr

    def test_multicondf_to_cr(self, graph_enabled):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @partial(
            qml.transforms.decompose,
            gate_set={hqml.ConditionalParity, qml.CNOT},
        )
        @qml.qnode(dev)
        def circuit():
            hqml.qcond(hqml.Fourier(0), [1, 2])

        tape = qml.workflow.construct_tape(circuit)()
        assert tape.operations == [
            qml.CNOT([1, 2]),
            hqml.ConditionalParity([2, 0]),
            qml.CNOT([1, 2]),
        ]

    def test_condbs_to_cbs(self, graph_enabled):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @partial(
            qml.transforms.decompose,
            gate_set={hqml.ConditionalBeamsplitter, qml.CNOT},
        )
        @qml.qnode(dev)
        def circuit():
            hqml.ops.QubitConditioned(hqml.Beamsplitter(0.5, 0, [0, 1]), 2)

        tape = qml.workflow.construct_tape(circuit)()
        assert tape.operations == [hqml.ConditionalBeamsplitter(0.5, 0, [2, 0, 1])]

    def test_multicondbs_to_cbs(self, graph_enabled):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @partial(
            qml.transforms.decompose,
            gate_set={hqml.ConditionalBeamsplitter, qml.CNOT},
        )
        @qml.qnode(dev)
        def circuit():
            hqml.ops.QubitConditioned(hqml.Beamsplitter(0.5, 0, [0, 1]), [2, 3])

        tape = qml.workflow.construct_tape(circuit)()
        assert tape.operations == [
            qml.CNOT([2, 3]),
            hqml.ConditionalBeamsplitter(0.5, 0, [3, 0, 1]),
            qml.CNOT([2, 3]),
        ]

    def test_cond_cd(self, graph_enabled):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @partial(
            qml.transforms.decompose,
            gate_set={hqml.ConditionalDisplacement, qml.CNOT},
        )
        @qml.qnode(dev)
        def circuit():
            hqml.qcond(hqml.ConditionalDisplacement(0.5, 0, [0, 1]), [2, 3])

        tape = qml.workflow.construct_tape(circuit)()
        assert tape.operations == [
            qml.CNOT([2, 3]),
            qml.CNOT([3, 0]),
            hqml.ConditionalDisplacement(0.5, 0, [0, 1]),
            qml.CNOT([3, 0]),
            qml.CNOT([2, 3]),
        ]
