# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape

import hybridlane as hqml
from hybridlane import io


def test_openqasm():
    def circuit(n):
        for j in range(n):
            qml.X(0)
            hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, 1])
            hqml.SelectiveNumberArbitraryPhase(0.5, j, [0, 1])

        return (
            hqml.expval(hqml.NumberOperator(1)),
            hqml.expval(qml.PauliZ(0)),
            hqml.expval(hqml.QuadP(1)),  # should be diagonalized
        )

    with QuantumTape() as tape:
        circuit(5)

    print(io.to_openqasm(tape, precision=5, strict=True))
