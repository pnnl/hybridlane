# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qp
import pytest
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane import sa


@pytest.mark.unit
class TestWireTypeChecking:
    @pytest.mark.parametrize("obs", (hl.NumberOperator, hl.QuadX))
    def test_no_operations(self, obs):
        # This tests that even with no operations, we can infer from the observables
        with qp.queuing.AnnotatedQueue() as q:
            hl.expval(obs(0) @ qp.PauliX(1))

        tape = QuantumScript.from_queue(q)
        res = sa.analyze(tape)
        assert res.qumodes == Wires(0)
        assert res.qubits == Wires(1)

    def test_some_operations(self):
        with qp.queuing.AnnotatedQueue() as q:
            hl.ConditionalDisplacement(0, 0, wires=[1, 0])
            qp.X(1)
            qp.Displacement(0, 0, wires=[0])

        tape = QuantumScript.from_queue(q)
        res = sa.analyze(tape)
        assert res.qumodes == Wires(0)
        assert res.qubits == Wires(1)
