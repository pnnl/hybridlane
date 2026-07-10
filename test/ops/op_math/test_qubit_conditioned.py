# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
from functools import partial

import pennylane as qp
import pytest

import hybridlane as hl
from hybridlane.ops import QubitConditioned


@pytest.mark.unit
class TestQubitConditioned:
    def test_name(self):
        op = QubitConditioned(hl.Rotation(0.5, 0), 1)
        assert op.name == "QubitConditioned(Rotation)"

    def test_overlapping_wires(self):
        with pytest.raises(ValueError):
            hl.qcond(qp.RZ(0.5, 0), 0)

    def test_known_gates(self):
        # Hybridlane gates
        assert hl.qcond(hl.Rotation(0.5, 0), 1) == hl.ConditionalRotation(1.0, [1, 0])
        assert hl.qcond(hl.Fourier(0), 1) == hl.ConditionalParity([1, 0])
        assert hl.qcond(hl.Beamsplitter(0.1, 0.2, [0, 1]), 2) == hl.ConditionalBeamsplitter(
            0.1, 0.2, [2, 0, 1]
        )
        assert hl.qcond(hl.Displacement(0.1, 0.2, 0), 1) == hl.ConditionalDisplacement(
            0.1, 0.2, [1, 0]
        )
        assert hl.qcond(hl.TwoModeSqueezing(0.1, 0, [0, 1]), 2) == hl.ConditionalTwoModeSqueezing(
            0.1, 0, [2, 0, 1]
        )

        # Pennylane gates
        assert hl.qcond(qp.GlobalPhase(0.5, 0), 1) == qp.RZ(1.0, 1)
        assert hl.qcond(qp.GlobalPhase(0.5, 0), [1, 2]) == qp.IsingZZ(1.0, [1, 2])
        assert hl.qcond(qp.IsingZZ(0.5, [0, 1]), [2, 3]) == qp.MultiRZ(0.5, [2, 3, 0, 1])

    def test_nested_qcond(self):
        op = hl.qcond(QubitConditioned(hl.Displacement(0.5, 0, 0), 1), 2)
        assert op == QubitConditioned(hl.Displacement(0.5, 0, 0), [2, 1])
        assert op.has_decomposition
        assert op.decomposition() == [
            qp.CNOT([2, 1]),
            hl.ConditionalDisplacement(0.5, 0, [1, 0]),
            qp.CNOT([2, 1]),
        ]


@pytest.mark.unit
class TestDecomposition:
    def test_rz_to_isingzz(self):
        op = QubitConditioned(qp.RZ(0.5, 0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [qp.IsingZZ(0.5, [1, 0])]

    def test_d_to_cd(self):
        op = QubitConditioned(hl.Displacement(0.1, 0.2, 0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [hl.ConditionalDisplacement(0.1, 0.2, [1, 0])]

    def test_f_to_cp(self):
        op = QubitConditioned(hl.Fourier(0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [hl.ConditionalParity([1, 0])]

    def test_r_to_cr(self):
        op = QubitConditioned(hl.Rotation(0.5, 0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [hl.ConditionalRotation(1.0, [1, 0])]

    def test_multirz(self):
        op = QubitConditioned(qp.MultiRZ(0.5, [0, 1]), [2, 3])
        assert op.has_decomposition
        assert op.decomposition() == [qp.MultiRZ(0.5, [2, 3, 0, 1])]

    def test_identity(self):
        op = QubitConditioned(qp.Identity(0), 1)
        assert op.has_decomposition
        assert op.decomposition() == [qp.Identity([1, 0])]

    def test_cnot_decomposition(self):
        op = QubitConditioned(hl.Displacement(0.1, 0.2, 0), [1, 2])
        assert op.has_decomposition
        assert op.decomposition() == [
            qp.CNOT([1, 2]),
            hl.qcond(hl.Displacement(0.1, 0.2, 0), 2),
            qp.CNOT([1, 2]),
        ]


@pytest.mark.usefixtures("enable_graph_decomp")
@pytest.mark.integration
class TestGraphDecomposition:
    def test_qcondf_to_cr(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalRotation},
        )
        @qp.qnode(dev)
        def circuit():
            hl.qcond(hl.Fourier(0), 1)

        tape = qp.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 1  # 1 cr

    def test_multi_qcondf_to_cr(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalRotation, qp.CNOT},
        )
        @qp.qnode(dev)
        def circuit():
            hl.qcond(hl.Fourier(0), [1, 2])

        tape = qp.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 3  # 1 cr, 2 cnot

    def test_ctrlf_to_cr(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalRotation, hl.Rotation, qp.CNOT},
        )
        @qp.qnode(dev)
        def circuit():
            qp.ctrl(hl.Fourier(0), 1)

        tape = qp.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 2  # 1 r, 1 cr

    def test_multicondf_to_cr(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalParity, qp.CNOT},
        )
        @qp.qnode(dev)
        def circuit():
            hl.qcond(hl.Fourier(0), [1, 2])

        tape = qp.workflow.construct_tape(circuit)()
        assert tape.operations == [
            qp.CNOT([1, 2]),
            hl.ConditionalParity([2, 0]),
            qp.CNOT([1, 2]),
        ]

    def test_condbs_to_cbs(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalBeamsplitter, qp.CNOT},
        )
        @qp.qnode(dev)
        def circuit():
            hl.ops.QubitConditioned(hl.Beamsplitter(0.5, 0, [0, 1]), 2)

        tape = qp.workflow.construct_tape(circuit)()
        assert tape.operations == [hl.ConditionalBeamsplitter(0.5, 0, [2, 0, 1])]

    def test_multicondbs_to_cbs(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalBeamsplitter, qp.CNOT},
        )
        @qp.qnode(dev)
        def circuit():
            hl.ops.QubitConditioned(hl.Beamsplitter(0.5, 0, [0, 1]), [2, 3])

        tape = qp.workflow.construct_tape(circuit)()
        assert tape.operations == [
            qp.CNOT([2, 3]),
            hl.ConditionalBeamsplitter(0.5, 0, [3, 0, 1]),
            qp.CNOT([2, 3]),
        ]

    def test_cond_cd(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalDisplacement, qp.CNOT},
        )
        @qp.qnode(dev)
        def circuit():
            hl.qcond(hl.ConditionalDisplacement(0.5, 0, [0, 1]), [2, 3])

        tape = qp.workflow.construct_tape(circuit)()
        assert tape.operations == [
            qp.CNOT([2, 3]),
            qp.CNOT([3, 0]),
            hl.ConditionalDisplacement(0.5, 0, [0, 1]),
            qp.CNOT([3, 0]),
            qp.CNOT([2, 3]),
        ]

    def test_cond_pow_cr(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalRotation, qp.CNOT},
        )
        @qp.qnode(dev)
        def circuit():
            hl.qcond(qp.pow(hl.ConditionalRotation(0.5, [0, 1]), 5), [2, 3])

        tape = qp.workflow.construct_tape(circuit)()
        assert tape.operations == [
            qp.CNOT([2, 3]),
            qp.CNOT([3, 0]),
            hl.ConditionalRotation(2.5, [0, 1]),
            qp.CNOT([3, 0]),
            qp.CNOT([2, 3]),
        ]

    def test_pow_adj_cr(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalRotation},
        )
        @qp.qnode(dev)
        def circuit():
            qp.pow(qp.adjoint(hl.ConditionalRotation(0.5, [0, 1])), 5)

        tape = qp.workflow.construct_tape(circuit)()
        assert tape.operations == [hl.ConditionalRotation(-2.5, [0, 1])]

    def test_pow_cond_cr(self):
        dev = qp.device("default.hybrid", fock_level=8)

        @partial(
            qp.transforms.decompose,
            gate_set={hl.ConditionalRotation, qp.CNOT},
        )
        @qp.qnode(dev)
        def circuit():
            qp.pow(hl.qcond(hl.ConditionalRotation(0.5, [0, 1]), [2, 3]), 5)

        tape = qp.workflow.construct_tape(circuit)()
        assert tape.operations == [
            qp.CNOT([2, 3]),
            qp.CNOT([3, 0]),
            hl.ConditionalRotation(2.5, [0, 1]),
            qp.CNOT([3, 0]),
            qp.CNOT([2, 3]),
        ]
