# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qp
import pytest
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane.wires import TypedWires, type_check


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
