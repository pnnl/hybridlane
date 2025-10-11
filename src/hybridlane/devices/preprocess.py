# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn

from .. import sa


@qml.transform
def static_analyze_tape(
    tape: QuantumScript, fill_missing: str | None = None
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Circuit pass that validates a wire is only used as a qubit or a qumode

    This validates that once a wire is used in an operation as a qubit or a qumode, that its
    role remains the same in the rest of the circuit. This is physically motivated since a device
    cannot perform swap gates between qubits and qumodes (there would be truncation issues).

    Example

    .. code:: python

        # This would throw an error
        def bad_circuit():
            qml.Displacement(alpha, 0, wires=[0])
            qml.X(1)
            qml.H(0) # error: wire 0 became a qumode earlier during Displacement

    Args:
        tape: The quantum circuit to check

        fill_missing: An optional string of ``("qubits", "qumodes")`` specifying what default
            to provide for unidentified wires

    Raises:
        :py:class:`~hybridlane.sa.StaticAnalysisError` if any wire is used as both a qubit and a qumode across the circuit, or
        if its type cannot be inferred and no default is provided.
    """

    sa.analyze(tape, fill_missing=fill_missing)  # errors if anything is wrong

    return (tape,), null_postprocessing


def null_postprocessing(results):
    return results[0]
