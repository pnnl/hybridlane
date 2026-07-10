# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Transforms and utilities for preprocessing quantum circuits before execution."""

from collections.abc import Hashable, Mapping

import pennylane as qp
from pennylane.devices.default_qubit import null_postprocessing
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn

from .. import wires as sa
from ..measurements.base import ShapeRequiresWireDims
from ..wires import Qubit, Qudit, Qumode


@qp.transform
def static_analyze_tape(
    tape: QuantumScript,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Circuit pass that validates a wire is only used as a qubit or a qumode

    This validates that once a wire is used in an operation as a qubit or a qumode, that its
    role remains the same in the rest of the circuit. This is physically motivated since a device
    cannot perform swap gates between qubits and qumodes (there would be truncation issues).

    Example:

    .. code:: python

        # This would throw an error
        def bad_circuit():
            qp.Displacement(alpha, 0, wires=[0])
            qp.X(1)
            qp.H(0) # error: wire 0 became a qumode earlier during Displacement

    Args:
        tape: The quantum circuit to check

    Raises:
        :py:class:`~hybridlane.sa.StaticAnalysisError` if any wire is used as both a qubit and a
            qumode across the circuit, or if its type cannot be inferred and no default is
            provided.
    """
    sa.type_check(tape)  # errors if anything is wrong

    return (tape,), null_postprocessing


@qp.transform
def fill_wire_dims(
    tape: QuantumScript,
    wire_dims: Mapping[Hashable, int] | None = None,
    default_qumode_dim: int | None = None,
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Transform pass that fills in missing wire dimensions for measurements

    To be compatible with JAX, certain measurements require the wire dimensions to be known at
    construction time so that JAX may deduce their shape statically. However, in hybridlane,
    truncations are set by the device and not the circuit definition, so this pass inserts the
    dimensions into the measurements after the circuit has been constructed.
    """
    wire_dims = wire_dims or {}
    wire_dims = dict(wire_dims)

    # First type check the circuit and fill any missing qumode dimensions
    res = sa.type_check(tape)
    for wire, type_ in res.wire_types.items():
        match type_:
            case Qubit():
                wire_dims.setdefault(wire, 2)
            case Qumode():
                wire_dims.setdefault(wire, default_qumode_dim)
            case Qudit(d):
                wire_dims.setdefault(wire, d)

    new_measurements = []
    for mp in tape.measurements:
        if isinstance(mp, ShapeRequiresWireDims):
            new_measurements.append(mp.copy_with_wire_dims(wire_dims))
        else:
            new_measurements.append(mp)

    new_tape = tape.copy(measurements=new_measurements)
    return (new_tape,), null_postprocessing
