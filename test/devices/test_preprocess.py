# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import pennylane as qml
import pytest
from pennylane.tape import QuantumScript

import hybridlane as hqml
from hybridlane.devices import preprocess
from hybridlane.sa.exceptions import StaticAnalysisError


class TestValidateWireTypes:
    def test_bad_circuit(self):
        with qml.queuing.AnnotatedQueue() as q:
            hqml.ConditionalDisplacement(0, 0, wires=[0, 1])
            qml.X(0)

        tape = QuantumScript.from_queue(q)

        with pytest.raises(StaticAnalysisError):
            preprocess.static_analyze_tape(tape)

    def test_good_circuit(self):
        with qml.queuing.AnnotatedQueue() as q:
            hqml.ConditionalDisplacement(0, 0, wires=[0, 1])
            qml.X(1)
            qml.Displacement(0, 0, wires=[0])

        tape = QuantumScript.from_queue(q)
        preprocess.static_analyze_tape(tape)
