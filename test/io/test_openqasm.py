import jax.numpy as jnp
import pennylane as qml
from pennylane.tape import QuantumTape

import hybridlane as hqml
from hybridlane import io


def test_openqasm():
    def circuit(n):
        for j in range(n):
            qml.X(0)
            hqml.JaynesCummings(jnp.pi / (2 * jnp.sqrt(j + 1)), jnp.pi / 2, [1, 0])
            hqml.SelectiveNumberArbitraryPhase(0.5, j, [1, 0])

        return (
            hqml.expval(hqml.NumberOperator(1)),
            hqml.expval(qml.PauliZ(0)),
            hqml.expval(hqml.QuadP(1)),  # should be diagonalized
        )

    with QuantumTape() as tape:
        circuit(5)

    print(io.to_openqasm(tape, precision=5, strict=True))
