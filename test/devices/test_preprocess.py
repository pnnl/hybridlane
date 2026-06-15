# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qp
import pytest
from pennylane.tape import QuantumScript

import hybridlane as hl
from hybridlane.devices import preprocess
from hybridlane.wires.exceptions import TypeCheckError


@pytest.mark.unit
class TestValidateWireTypes:
    def test_bad_circuit(self):
        with qp.queuing.AnnotatedQueue() as q:
            hl.ConditionalDisplacement(0, 0, wires=[1, 0])
            qp.X(0)

        tape = QuantumScript.from_queue(q)

        with pytest.raises(TypeCheckError):
            preprocess.static_analyze_tape(tape)

    def test_good_circuit(self):
        with qp.queuing.AnnotatedQueue() as q:
            hl.ConditionalDisplacement(0, 0, wires=[1, 0])
            qp.X(1)
            qp.Displacement(0, 0, wires=[0])

        tape = QuantumScript.from_queue(q)
        preprocess.static_analyze_tape(tape)
