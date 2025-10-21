# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
r"""Module containing export functions to a superset of OpenQASM

Here we give an example with a circuit that performs a measurement that is non-diagonal (because
the computational basis in phase space is :math:`\hat{x}`).

.. code:: python

    import pennylane as qml
    import hybridlane as hqml

    dev = qml.device("bosonicqiskit.hybrid")

    @qml.qnode(dev)
    def circuit(n):
        for j in range(n):
            qml.X(0)
            hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, 1])

        return (
            hqml.var(hqml.QuadP(1)),
            hqml.expval(qml.PauliZ(0)),
        )

    qasm = hqml.to_openqasm(circuit, precision=5, strict=strict)(5)

Output with ``strict=False`` (the default)

.. code::

    OPENQASM 3.0;
    include "stdgates.inc";

    const int homodyne_precision_bits = 32;
    const int fock_readout_precision_bits = 32;
    include "cvstdgates.inc";


    qubit[1] q;
    qumode[1] m;

    def state_prep() {
        reset q;
        reset m;
        x q[0];
        cv_jc(1.5708, 1.5708) q[0], m[0];
        x q[0];
        cv_jc(1.1107, 1.5708) q[0], m[0];
        x q[0];
        cv_jc(0.9069, 1.5708) q[0], m[0];
        x q[0];
        cv_jc(0.7854, 1.5708) q[0], m[0];
        x q[0];
        cv_jc(0.70248, 1.5708) q[0], m[0];
    }

    state_prep();
    cv_r(-1.5708) m[0];
    float[homodyne_precision_bits] c0 = measure_x m[0];
    bit[1] c1;
    c1[0] = measure q[0];

Output with ``strict=True``

.. code::

    OPENQASM 3.0;
    include "stdgates.inc";

    const int homodyne_precision_bits = 32;
    const int fock_readout_precision_bits = 32;
    include "cvstdgates.inc";


    // Position measurement x
    defcal measure_x m -> float[homodyne_precision_bits] {}

    // Fock measurement n
    defcal measure_n m -> uint[fock_readout_precision_bits] {}

    qubit[1] q;
    qubit[1] m;

    def state_prep() {
        reset q;
        reset m;
        x q[0];
        cv_jc(1.5708, 1.5708) q[0], m[0];
        x q[0];
        cv_jc(1.1107, 1.5708) q[0], m[0];
        x q[0];
        cv_jc(0.9069, 1.5708) q[0], m[0];
        x q[0];
        cv_jc(0.7854, 1.5708) q[0], m[0];
        x q[0];
        cv_jc(0.70248, 1.5708) q[0], m[0];
    }

    state_prep();
    cv_r(-1.5708) m[0];
    float[homodyne_precision_bits] c0 = measure_x(m[0]);
    bit[1] c1;
    c1[0] = measure q[0];
"""

import textwrap
from functools import wraps
from typing import Any, Callable

import pennylane as qml
from pennylane.io.to_openqasm import OPENQASM_GATES
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.tape import QuantumScript
from pennylane.wires import Wires

import hybridlane as hqml

from .. import sa
from ..transforms import from_pennylane


def to_openqasm(
    qnode,
    rotations: bool = True,
    precision: int | None = None,
    strict: bool = False,
    float_bits: int = 32,
    int_bits: int = 32,
    indent: int = 4,
    level: str | None = "user",
) -> Callable[[Any], str]:
    r"""Converts a circuit to an OpenQASM 3.0 program

    By default, the output will be a *superset* of the OpenQASM standard with extra features and language
    extensions that capture hybrid CV-DV programs. These modifications are detailed in the documentation.
    If you would like the output to be strictly compliant with OpenQASM 3.0, you can pass the ``strict=True``
    flag, which will

    1. Replace ``measure_x`` and ``measure_n`` keywords with equivalent ``defcal`` statements and function calls.

    2. Remove all ``qumode`` keywords, replacing them with ``qubit``. This has the effect of erasing the type information of the program.

    .. note::
        Qubit measurements are assumed to be performed in the computational basis, while
        qumode measurements are determined from the :class:`~hybridlane.sa.base.BasisSchema` of each
        measurement. If sampling an observable, this function can provide the gates necessary to diagonalize
        each observable by setting ``rotations=True``. Only wires that are actually measured will have measurement
        statements. Finally, non-overlapping measurements will be grouped together as much as possible and
        measured on the same call to ``state_prep()``; however, the resulting program may have multiple executions
        of the tape as needed to accomodate all the measurements.

    Args:
        qnode: The QNode to be converted to OpenQASM

        rotations: Include diagonalizing gates for an observable prior to measurement. This applies
            both to qubit observables and qumode observables.

        precision: An optional number of decimal places to use when recording the angle parameters of each gate

        strict: Forces the output to be strictly compliant with the OpenQASM 3.0 parser.

        float_bits: The number of bits used to contain the result of a homodyne measurement.

        int_bits: The number of bits used to contain the result of a Fock state readout.

        indent: Number of spaces to indent the program by

    Returns:
        A string containing the program in OpenQASM 3.0
    """
    from pennylane.workflow import construct_tape

    @wraps(qnode)
    def wrapper(*args, **kwargs) -> str:
        tape = construct_tape(qnode, level="user")(*args, **kwargs)
        (tape,), _ = from_pennylane(tape)  # compatibility with pl gates
        return tape_to_openqasm(
            tape,
            rotations=rotations,
            precision=precision,
            strict=strict,
            float_bits=float_bits,
            int_bits=int_bits,
            indent=indent,
        )

    return wrapper


###########################################
#           Gate definitions
###########################################

# CV "standard library", included in "cvstdgates.inc"
cv_stdgates: dict[str, str] = {
    "Rotation": "cv_r",
    "Displacement": "cv_d",
    "Squeezing": "cv_sq",
    "Kerr": "cv_k",
    "CubicPhase": "cv_c",
    "Fourier": "cv_f",
    "Beamsplitter": "cv_bs",
    "TwoModeSqueezing": "cv_tms",
    "TwoModeSum": "cv_sum",
    "ConditionalRotation": "cv_cr",
    "ConditionalDisplacement": "cv_cd",
    "ConditionalSqueezing": "cv_cs",
    "ConditionalParity": "cv_cp",
    "SelectiveQubitRotation": "cv_sqr",
    "SelectiveNumberArbitraryPhase": "cv_snap",
    "ModeSwap": "cv_swap",
    "JaynesCummings": "cv_jc",
    "AntiJaynesCummings": "cv_ajc",
    "ConditionalBeamsplitter": "cv_cbs",
    "ConditionalTwoModeSqueezing": "cv_ctms",
    "ConditionalTwoModeSum": "cv_csum",
}

all_gates = OPENQASM_GATES | cv_stdgates


# These are our special extensions to OpenQASM
class Keywords:
    CvStdLib = "cvstdgates.inc"
    QumodeDef = "qumode"
    MeasureQuadX = "measure_x"
    MeasureN = "measure_n"
    HomodynePrecision = "homodyne_precision_bits"
    FockPrecision = "fock_readout_precision_bits"


def get_header(float_bits: int, int_bits: int):
    return textwrap.dedent(f"""
        OPENQASM 3.0;
        include "stdgates.inc";

        const int {Keywords.HomodynePrecision} = {float_bits};
        const int {Keywords.FockPrecision} = {int_bits};
        include "{Keywords.CvStdLib}";
        """)


# We leave the calibration bodies {} empty because they should be opaque definitions.
# In principle, these could be hardware pulse definitions.
def get_cv_calibration_definition():
    return textwrap.dedent(f"""
        // Position measurement x
        defcal {Keywords.MeasureQuadX} m -> float[{Keywords.HomodynePrecision}] {{}}

        // Fock measurement n
        defcal {Keywords.MeasureN} m -> uint[{Keywords.FockPrecision}] {{}}""")


# This version only unrolls all the gates. A more advanced version that captures the loop
# and conditional branching structure would require plxpr
def tape_to_openqasm(
    tape: QuantumScript,
    rotations: bool = True,
    precision: int | None = None,
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

    qasm_str = get_header(float_bits, int_bits) + "\n"

    if strict:
        qasm_str += get_cv_calibration_definition() + "\n"

    qasm_str += "\n"

    # For strict compliance with openqasm, call all qumodes "qubits", losing
    # the ability to verify types easily
    if res.qubits:
        qasm_str += f"qubit[{len(res.qubits)}] q;\n"

    if res.qumodes:
        kw = "qubit" if strict else Keywords.QumodeDef
        qasm_str += f"{kw}[{len(res.qumodes)}] m;\n"

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
                qasm_str += f"bit[{len(measured_qubits)}] {cvar};\n"
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
                        result_type, func = (
                            f"uint[{Keywords.FockPrecision}]",
                            Keywords.MeasureN,
                        )
                    elif basis == sa.ComputationalBasis.Position:
                        result_type, func = (
                            f"float[{Keywords.HomodynePrecision}]",
                            Keywords.MeasureQuadX,
                        )
                    else:
                        raise ValueError("Unsupported basis", basis)

                    if strict:
                        qasm_str += (
                            f"{result_type} {cvar} = {func}({wire_to_str[qumode]});\n"
                        )
                    else:
                        qasm_str += (
                            f"{result_type} {cvar} = {func} {wire_to_str[qumode]};\n"
                        )

        qasm_str += "\n"

    return qasm_str


def _format_gate(
    op: Operator, wire_to_str: dict[Any, str], precision: int | None = None
) -> str:
    if (gate_name := all_gates.get(op.name)) is None:
        raise ValueError(f"Unsupported gate {op.name}")

    if precision:
        params = list(map(lambda p: f"{p:.{precision}f}", op.parameters))
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
