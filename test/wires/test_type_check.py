# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qp
import pytest
from pennylane.operation import Operator
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane.measurements import ComputationalBasis
from hybridlane.wires import Qubit, TypedWires, infer_measurement_bases, type_check
from hybridlane.wires.type_check import infer_wires


@pytest.mark.unit
class TestWireTypeChecking:
    @pytest.mark.parametrize("obs", (hl.NumberOperator, hl.QuadX))
    def test_no_operations(self, obs):
        # This tests that even with no operations, we can infer from the observables
        with qp.queuing.AnnotatedQueue() as q:
            hl.expval(obs(0) @ qp.PauliX(1))

        tape = QuantumScript.from_queue(q)
        res = type_check(tape)
        assert res.qumodes == Wires(0)
        assert res.qubits == Wires(1)

    def test_some_operations(self):
        with qp.queuing.AnnotatedQueue() as q:
            hl.ConditionalDisplacement(0, 0, wires=[1, 0])
            qp.X(1)
            qp.Displacement(0, 0, wires=[0])

        tape = QuantumScript.from_queue(q)
        res = type_check(tape)
        assert res.qumodes == Wires(0)
        assert res.qubits == Wires(1)

    def test_with_registers(self):
        reg = hl.qumodes({"a": 3})
        assert isinstance(reg["a"], TypedWires)

        wires = reg["a"]
        tape = QuantumScript(
            [
                qp.I(wires[0]),
                qp.I(wires[1]),
                qp.I(wires[2]),
            ]
        )  # ops that carry no type information
        res = type_check(tape)
        assert len(res.qumodes) == 3

    @pytest.mark.parametrize(
        "op",
        [
            qp.H(0),
            qp.X(0),
            qp.CNOT((0, 1)),
            qp.Projector([1], wires="a"),
            qp.Projector([1, 0, 1], wires=(0, 1, 2)),
            qp.MultiRZ(0.1, wires=range(10)),
            qp.Hermitian([[1, 0], [0, 1]], wires=1),
        ],
    )
    def test_with_qubit_ops(self, op: Operator):
        context = infer_wires(op, {})
        assert all(w in context for w in op.wires)
        assert all(context[w] == Qubit() for w in op.wires)


@pytest.mark.unit
class TestInferMeasurementBases:
    @pytest.mark.parametrize(
        "obs",
        [
            qp.Projector([1], wires="a"),
            qp.Projector([1, 0, 1], wires=(0, 1, 2)),
            qp.Hermitian([[1, 0], [0, 1]], wires=1),
            qp.Z(0),
            qp.X(0),
            qp.Z(0) @ qp.X(1),
            qp.X(0) + 0.5 * qp.Z(2),
        ],
    )
    def test_qubit_observables(self, obs):
        bases = infer_measurement_bases(obs, {})
        assert all(bases[w] == ComputationalBasis.Discrete for w in obs.wires)
