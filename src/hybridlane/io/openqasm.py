"""Module containing export functions to a superset of OpenQASM"""

from typing import Any, Optional

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

import hybridlane as hqml

from .. import sa

###########################################
#           Gate definitions
###########################################

# Standard QASM library, see https://openqasm.com/language/standard_library.html
openqasm_stdgates: dict[str, str] = {
    "GlobalPhase": "gphase",
    "Identity": "id",
    "Hadamard": "h",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "S": "s",
    "T": "t",
    # Somehow we need sdg and tdg too
    "SX": "sx",
    "Rot": "u",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "PhaseShift": "p",
    "ControlledPhaseShift": "cp",
    "CNOT": "cx",
    "CZ": "cz",
    "CY": "cy",
    "CH": "ch",
    "SWAP": "swap",
    "CSWAP": "cswap",
    "Toffoli": "ccx",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
}

# CV "standard library", taken from Table IV.4 of https://arxiv.org/abs/2407.10381
cv_stdgates: dict[str, str] = {
    # Gaussian operations
    "Displacement": "cv_d",
    "Rotation": "cv_r",
    "Squeezing": "cv_sq",
    "Beamsplitter": "cv_bs",
    "TwoModeSqueezing": "cv_sq2",
    # Cubic ISA
    "CubicPhase": "cv_p3",
    # SNAP ISA
    "SelectiveNumberArbitraryPhase": "cv_snap",
    # Phase-space ISA
    "ConditionalDisplacement": "cv_cd",
    # Fock-space ISA
    "SelectiveQubitRotation": "cv_sqr",
    # Sideband ISA
    "JaynesCummings": "cv_jc",
    # Convenience
    "ModeSwap": "cv_swap",
    "Fourier": "cv_f",
    "AntiJaynesCummings": "cv_ajc",
    "ConditionalParity": "cv_cp",
    # Controlled variants of above gates (maybe needed for `ctrl @ gate` syntax)
    "ConditionalRotation": "cv_cr",
    "ConditionalBeamsplitter": "cv_cbs",
    "ConditionalTwoModeSqueezing": "cv_csq2",
}

all_gates = openqasm_stdgates | cv_stdgates


# These are our special extensions to OpenQASM
class Keywords:
    CvStdLib = "cvstdgates.inc"
    QumodeDef = "qumode"
    MeasureQuadX = "homodyne"
    MeasureN = "fock_number"


HEADER = f"""
OPENQASM 3.0;

include "stdgates.inc";
include "{Keywords.CvStdLib}";
""".strip()


# We leave the calibration bodies {} empty because they should be opaque definitions.
# In principle, these could be hardware pulse definitions.
def get_cv_calibration_definition(
    strict: bool = False,
    float_bits: int = 32,
    int_bits: int = 32,
):
    kw = "qubit" if strict else Keywords.QumodeDef
    return f"""
cal {{
    // Position measurement x
    defcal {Keywords.MeasureQuadX}({kw} q) -> float[{float_bits}] {{}}

    // Fock measurement n
    defcal {Keywords.MeasureN}({kw} q) -> uint[{int_bits}] {{}}
}}
"""


# This version only unrolls all the gates. A more advanced version that captures the loop
# and conditional branching structure would require plxpr
def to_openqasm(
    tape: QuantumScript,
    rotations: bool = True,
    precision: Optional[int] = None,
    strict: bool = False,
    float_bits: int = 32,
    int_bits: int = 32,
    indent: int = 4,
):
    # Preprocessing
    tape = tape.map_to_standard_wires()
    [tape], _ = qml.transforms.convert_to_numpy_parameters(tape)
    res = sa.analyze(tape)

    wire_to_str = {w: f"q[{i}]" for i, w in enumerate(res.qubits)} | {
        w: f"m[{i}]" for i, w in enumerate(res.qumodes)
    }

    qasm_str = f"{HEADER}\n\n"
    # For strict compliance with openqasm, call all qumodes "qubits", losing
    # the ability to verify types easily
    if res.qubits:
        qasm_str += f"qubit q[{len(res.qubits)}];\n"

    if res.qumodes:
        kw = "qubit" if strict else Keywords.QumodeDef
        qasm_str += f"{kw} m[{len(res.qumodes)}];\n"

    qasm_str += get_cv_calibration_definition(
        strict=strict, float_bits=float_bits, int_bits=int_bits
    )

    # Construct the state prep function consisting of all the circuit gates
    # prior to the measurements
    qasm_str += "\ndef state_prep() {\n"

    if res.qubits:
        qasm_str += " " * indent + "reset q;\n"
    if res.qumodes:
        qasm_str += " " * indent + "reset m;\n"

    just_ops = QuantumScript(tape.operations)
    operations = just_ops.expand(
        depth=10, stop_at=lambda op: op.name in all_gates
    ).operations
    for op in operations:
        qasm_str += (
            " " * indent + _format_gate(op, wire_to_str, precision=precision) + "\n"
        )

    qasm_str += "}\n"

    # Now identify the minimal groups of measurements that can be performed together
    # on the same circuit. Note this is a special case of more general commuting observables
    measurement_groups: list[list[MeasurementProcess]] = []
    for mp in tape.measurements:
        found = False
        for group in measurement_groups:
            # If we find a non-overlapping measurement group, add this to it
            overlapping = Wires.shared_wires(
                [mp.wires, Wires.all_wires([m.wires for m in group])]
            )
            if not overlapping:
                group.append(mp)
                found = True
                continue

        # No group found
        if not found:
            measurement_groups.append([mp])

    qasm_str += "\n"
    classical_vars = 0
    for group in measurement_groups:
        qasm_str += "state_prep();\n"

        # Apply diagonalizing gates if the user requested it
        if rotations:
            for mp in group:
                operations = QuantumScript(mp.diagonalizing_gates())
                operations = operations.expand(
                    depth=10, stop_at=lambda op: op.name in all_gates
                ).operations

                for op in operations:
                    qasm_str += (
                        _format_gate(op, wire_to_str, precision=precision) + "\n"
                    )

        # Now measure, determining the appropriate measure function for each process
        for mp in group:
            all_wires = mp.wires
            measured_qubits = Wires(sorted(res.qubits & all_wires))

            # Qubits always get measured in z basis with <bit var> = measure <qubit> syntax
            if measured_qubits:
                cvar = f"c{classical_vars}"
                classical_vars += 1
                qasm_str += f"bit {cvar}[{len(measured_qubits)}]\n"
                for i, w in enumerate(measured_qubits):
                    qasm_str += f"{cvar}[{i}] = measure {wire_to_str[w]};\n"

            # Qumodes are more complicated, as we must determine whether it's a homodyne or fock measurement
            # from the basis schema
            if schema := res.schemas[tape.measurements.index(mp)]:
                measured_qumodes = res.qumodes & all_wires
                for qumode in measured_qumodes:
                    cvar = f"c{classical_vars}"
                    classical_vars += 1
                    basis = schema.get_basis(qumode)

                    if basis == sa.ComputationalBasis.Discrete:
                        result_type, func = f"uint[{int_bits}]", Keywords.MeasureN
                    elif basis == sa.ComputationalBasis.Position:
                        result_type, func = (
                            f"float[{float_bits}]",
                            Keywords.MeasureQuadX,
                        )
                    else:
                        raise ValueError("Unsupported basis", basis)

                    qasm_str += f"{result_type} {cvar};\n"
                    qasm_str += f"{cvar} = {func}({wire_to_str[qumode]});\n"

        qasm_str += "\n"

    return qasm_str


def _format_gate(
    op: Operator, wire_to_str: dict[Any, str], precision: Optional[int] = None
) -> str:
    if (gate_name := all_gates.get(op.name)) is None:
        raise ValueError(f"Unsupported gate {op.name}")

    if precision:
        params = list(map(lambda p: f"{p:.{precision}}", op.parameters))
    else:
        params = list(map(str, op.parameters))

    # Throw special exceptions in here
    # Todo: Check the convention of each pennylane/hybridlane gate
    # to its QASM definition
    match op:
        # Extract the fock level hyperparameter
        case (
            hqml.SelectiveNumberArbitraryPhase(hyperparameters=h)
            | hqml.SelectiveQubitRotation(hyperparameters=h)
        ):
            fock_level = h["n"]
            params.append(f"{fock_level:d}")

    wires = list(map(lambda w: wire_to_str[w], op.wires))
    param_str = "(" + ", ".join(params) + ")" if params else ""
    wire_str = ", ".join(wires)
    gate_str = f"{gate_name}{param_str} {wire_str};"
    return gate_str
