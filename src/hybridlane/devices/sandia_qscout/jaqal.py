# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

r"""Module for exporting circuits compiled to the ion trap to the Jaqal format"""

import re
from functools import singledispatch, wraps

import pennylane as qml
from pennylane.exceptions import DeviceError
from pennylane.operation import Operation
from pennylane.tape import QuantumScript

import hybridlane as hqml

from ... import sa
from . import ops as native_ops

# Mappings from the names of gates to Jaqal
# Obtainable from https://gitlab.com/jaqal/qscout-gatemodels/-/blob/master/src/qscout/v1/std/jaqal_gates.py?ref_type=heads
QUBIT_GATES = {
    "GlobalPhase": None,
    # "I": None,
    "R": "R",
    "RX": "Rx",
    "RY": "Ry",
    "RZ": "Rz",
    "PauliX": "Px",
    "PauliY": "Py",
    "PauliZ": "Pz",
    "SX": "Sx",
    "Adjoint(SX)": "Sxd",
    # No Sy in pennylane
    "S": "Sz",
    "Adjoint(S)": "Szd",
    "IsingXX": "XX",  # MS = Rxx(π/2)
    "IsingYY": "YY",
    "IsingZZ": "ZZ",
}

# Taken from the slides
BOSON_GATES = {
    "JaynesCummings": "Red",
    "AntiJaynesCummings": "Blue",
    "FockStatePrep": "FockState",
    "ConditionalXDisplacement": "SDF",
    "ConditionalXSqueezing": "RampUp",
    "NativeBeamsplitter": "Beamsplitter",
    "SidebandProbe": "Rt_SBProbe",
}


def to_jaqal(
    qnode, level: str | int | slice | None = None, precision: float | None = None
):
    from pennylane.workflow import construct_tape

    @wraps(qnode)
    def wrapper(*args, **kwargs) -> str:
        tape = construct_tape(qnode, level=level)(*args, **kwargs)
        return tape_to_jaqal(
            tape,
            precision=precision,
        )

    return wrapper


def tape_to_jaqal(tape: QuantumScript, precision: float | None = None):
    sa_res = sa.analyze(tape)
    num_qubits = max(sa_res.qubits) + 1

    program = f"register q[{num_qubits}]\n\n"
    program += "prepare_all\n"
    for op in tape.operations:
        program += tokenize_operation(op, precision=precision) + "\n"

    program += "measure_all"
    return program


@singledispatch
def tokenize_operation(op: Operation, precision: float | None = None) -> str:
    if gate_id := QUBIT_GATES.get(op.name, None):
        params = _tokenize_params(op.parameters, precision=precision)
        wires = [f"q[{w}]" for w in op.wires]
        return " ".join(map(str, [gate_id, *wires, *params]))

    raise DeviceError(f"Cannot serialize non-native gate to Jaqal: {op}")


@tokenize_operation.register
def _(op: native_ops.R, precision: int | None = None):
    gate_id = QUBIT_GATES[op.name]
    angle, axis_angle = _tokenize_params(op.parameters, precision=precision)
    qubit = op.wires[0]

    return f"{gate_id} q[{qubit}] {axis_angle} {angle}"


@tokenize_operation.register
def _(_op: qml.GlobalPhase | qml.Identity, **_):
    return ""


@tokenize_operation.register
def _(op: hqml.JaynesCummings, precision: float | None = None):
    # Has format Red <qubit> <Fock state> <phase> <angle> <axis choice> <mode choice>
    gate_id = BOSON_GATES[op.name]
    params = _tokenize_params(op.parameters, precision=precision)
    qubit, mode = op.wires
    axis, mode_idx = _get_axis_and_index(mode)

    # Hard code - the user has to compensate for the calibration at the moment
    fock_state = 1

    return (
        f"{gate_id} q[{qubit}] {fock_state} {params[1]} {params[0]} {axis} {mode_idx}"
    )


@tokenize_operation.register
def _(op: hqml.AntiJaynesCummings, precision: float | None = None):
    # Has format Blue <qubit> <Fock state> <phase> <angle> <axis choice> <mode choice>
    gate_id = BOSON_GATES[op.name]
    params = _tokenize_params(op.parameters, precision=precision)
    qubit, mode = op.wires
    axis, mode_idx = _get_axis_and_index(mode)

    # Hard code - the user has to compensate for the calibration at the moment
    fock_state = 1

    return (
        f"{gate_id} q[{qubit}] {fock_state} {params[1]} {params[0]} {axis} {mode_idx}"
    )


@tokenize_operation.register
def _(op: native_ops.FockStatePrep, **_):
    # Has format FockState <qubit> <fock state> <axis>
    gate_id = BOSON_GATES[op.name]
    fock_state = int(op.parameters[0])
    qubit, mode = op.wires
    axis, mode_idx = _get_axis_and_index(mode)

    return f"{gate_id} q[{qubit}] {fock_state} {axis}"


@tokenize_operation.register
def _(op: native_ops.SidebandProbe, precision: float | None = None):
    # Has format Rt_SBProbe <qubit> <phase> <duration_us> <axis> <mode> <sign> <detuning>
    gate_id = BOSON_GATES[op.name]
    [duration_us, phase, detuning] = _tokenize_params(
        [op.parameters[i] for i in (0, 1, 3)], precision=precision
    )
    sign = f"{op.parameters[2]:d}"
    qubit, mode = op.wires
    axis, mode_idx = _get_axis_and_index(mode)

    return f"{gate_id} q[{qubit}] {phase} {duration_us} {axis} {mode_idx} {sign} {detuning}"


@tokenize_operation.register
def _(op: native_ops.ConditionalXDisplacement, precision: float | None = None):
    # Has format SDF <qubit> <beta_re> <beta_im>
    gate_id = BOSON_GATES[op.name]
    params = _tokenize_params(op.parameters, precision=precision)
    qubit, mode = op.wires
    axis, mode_idx = _get_axis_and_index(mode)

    return f"{gate_id} q[{qubit}] {params[0]} {params[1]}"


@tokenize_operation.register
def _(op: native_ops.ConditionalXSqueezing, precision: float | None = None):
    # Has format RampUp <qubit> <blue/red ratio>
    gate_id = BOSON_GATES[op.name]
    params = _tokenize_params(op.parameters, precision=precision)
    qubit, mode = op.wires
    axis, mode_idx = _get_axis_and_index(mode)

    return f"{gate_id} q[{qubit}] {params[0]}"


@tokenize_operation.register
def _(op: native_ops.NativeBeamsplitter, precision: float | None = None):
    # Has format Beamsplitter <qubit> <detuning1> <detuning2> <duration> <phase>
    gate_id = BOSON_GATES[op.name]
    params = _tokenize_params(op.parameters, precision=precision)
    qubit, *modes = op.wires
    # axis, mode_idx = _get_axis_and_index(mode)

    return f"{gate_id} q[{qubit}] {params[0]} {params[1]} {params[2]} {params[3]}"


def _tokenize_params(params, precision: float | None = None):
    if precision:
        params = list(map(lambda p: f"{p:.{precision}}", params))
    else:
        params = list(map(str, params))

    return params


# Should match layout function in the main device
qumode_pattern = re.compile(r"^a([01])m(\d+)$")


def _get_axis_and_index(qumode):
    if isinstance(qumode, str) and (match := qumode_pattern.match(qumode)):
        return int(match.group(1)), int(match.group(2))

    raise DeviceError(f"Unsupported qumode name: {qumode}")
