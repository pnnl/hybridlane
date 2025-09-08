# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import pennylane as qml
import pytest
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

import hybridlane as hqml
from hybridlane import sa


class TestWireTypeChecking:
    @pytest.mark.parametrize("obs", (hqml.NumberOperator, hqml.QuadX))
    def test_no_operations(self, obs):
        # This tests that even with no operations, we can infer from the observables
        with qml.queuing.AnnotatedQueue() as q:
            hqml.expval(obs(0) @ qml.PauliX(1))

        tape = QuantumScript.from_queue(q)
        res = sa.analyze(tape)
        assert res.qumodes == Wires(0)
        assert res.qubits == Wires(1)

    def test_some_operations(self):
        with qml.queuing.AnnotatedQueue() as q:
            hqml.ConditionalDisplacement(0, 0, wires=[0, 1])
            qml.X(1)
            qml.Displacement(0, 0, wires=[0])

        tape = QuantumScript.from_queue(q)
        res = sa.analyze(tape)
        assert res.qumodes == Wires(0)
        assert res.qubits == Wires(1)
