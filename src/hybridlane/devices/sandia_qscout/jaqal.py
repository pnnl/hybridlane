# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

r"""Module for exporting circuits compiled to the ion trap to the Jaqal format"""

from __future__ import annotations

import decimal
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import singledispatch, wraps
from typing import TYPE_CHECKING, Literal, LiteralString, cast
from unittest.mock import patch

import pennylane as qml
from pennylane.exceptions import DeviceError
from pennylane.operation import Operation
from pennylane.tape import QuantumScript

import hybridlane as hqml

from ... import sa
from . import ops as native_ops

if TYPE_CHECKING:
    from jaqalpaq.qsyntax import qsyntax

QUBIT_BOSON_MODULE = "Calibration_PulseDefinitions.QubitBosonPulses"


@dataclass(frozen=True)
class Qumode:
    """Hardware qumode on the ion trap device."""

    manifold: Literal[0, 1]
    """The 'manifold' index dictating the axial direction

    The lower manifold is ``1`` with stronger coupling, while the upper manifold is
    ``0``.
    """

    index: int
    """Index of the qumode within the manifold, ranging from ``[0, n-1]``

    The modes are as follows:
        - ``0``: Center of Mass (COM) mode
        - ``1``: Tilt mode
        - ``2``: Drum/trapezoid mode
        - ``n-1``: Zig-zag mode
    """

    def __str__(self):
        return f"m{self.manifold}i{self.index}"

    @classmethod
    def try_from_string(cls, s: str):
        match = re.match(r"^m([01])i(\d+)$", s)
        if not match:
            return None

        manifold = int(match.group(1))
        index = int(match.group(2))
        return cls(manifold=manifold, index=index)


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
    "JaynesCummings": "JC",
    "AntiJaynesCummings": "AJC",
    "FockState": "FockStatePrep",
    "ConditionalDisplacement": "zCD",
    "ConditionalYDisplacement": "yCD",
    "ConditionalXDisplacement": "xCD",
    "ConditionalXSqueezing": "RampUp",
    "NativeBeamsplitter": "Beamsplitter",
    "SidebandProbe": "Rt_SBProbe",
}


# put in function so that it is not executed on import
def get_boson_gate_defs():
    from jaqalpaq.core import GateDefinition, Parameter, ParamType

    return {
        "prepare_all": GateDefinition("prepare_all", []),
        "measure_all": GateDefinition("measure_all", []),
        "JC": GateDefinition(
            "JC",
            parameters=[
                Parameter("qubit", ParamType.QUBIT),
                Parameter("manifold", ParamType.INT),
                Parameter("mode", ParamType.INT),
                Parameter("phase", ParamType.FLOAT),
                Parameter("angle", ParamType.FLOAT),
            ],
        ),
        "AJC": GateDefinition(
            "AJC",
            parameters=[
                Parameter("qubit", ParamType.QUBIT),
                Parameter("manifold", ParamType.INT),
                Parameter("mode", ParamType.INT),
                Parameter("phase", ParamType.FLOAT),
                Parameter("angle", ParamType.FLOAT),
            ],
        ),
        "FockStatePrep": GateDefinition(
            "FockStatePrep",
            parameters=[
                Parameter("q", ParamType.QUBIT),
                Parameter("manifold", ParamType.INT),
                Parameter("mode", ParamType.INT),
                Parameter("state", ParamType.INT),
            ],
        ),
        "xCD": GateDefinition(
            "xCD",
            parameters=[
                Parameter("q", ParamType.QUBIT),
                Parameter("manifold", ParamType.INT),
                Parameter("mode", ParamType.INT),
                Parameter("beta_re", ParamType.FLOAT),
                Parameter("beta_im", ParamType.FLOAT),
            ],
        ),
        "yCD": GateDefinition(
            "yCD",
            parameters=[
                Parameter("q", ParamType.QUBIT),
                Parameter("manifold", ParamType.INT),
                Parameter("mode", ParamType.INT),
                Parameter("beta_re", ParamType.FLOAT),
                Parameter("beta_im", ParamType.FLOAT),
            ],
        ),
        "zCD": GateDefinition(
            "zCD",
            parameters=[
                Parameter("q", ParamType.QUBIT),
                Parameter("manifold", ParamType.INT),
                Parameter("mode", ParamType.INT),
                Parameter("beta_re", ParamType.FLOAT),
                Parameter("beta_im", ParamType.FLOAT),
            ],
        ),
        "RampUp": GateDefinition(
            "RampUp",
            parameters=[
                Parameter("q", ParamType.QUBIT),
                Parameter("blue_red_ratio", ParamType.FLOAT),
            ],
        ),
        "Beamsplitter": GateDefinition(
            "Beamsplitter",
            parameters=[
                Parameter("q", ParamType.QUBIT),
                Parameter("detuning1", ParamType.FLOAT),
                Parameter("detuning2", ParamType.FLOAT),
                Parameter("duration", ParamType.FLOAT),
                Parameter("phase", ParamType.FLOAT),
            ],
        ),
        "Rt_SBProbe": GateDefinition(
            "Rt_SBProbe",
            parameters=[
                Parameter("q", ParamType.QUBIT),
                Parameter("phase", ParamType.FLOAT),
                Parameter("duration_us", ParamType.FLOAT),
                Parameter("manifold", ParamType.INT),
                Parameter("mode", ParamType.INT),
                Parameter("sign", ParamType.INT),
                Parameter("detuning", ParamType.FLOAT),
            ],
        ),
    }


def to_jaqal(qnode, level: str | int | slice = "user", precision: int = 20):
    @wraps(qnode)
    def wrapper(*args, **kwargs) -> str:
        batch, fn = qml.workflow.construct_batch(qnode, level=level)(*args, **kwargs)
        return batch_to_jaqal(
            batch,
            precision=precision,
        )

    return wrapper


def batch_to_jaqal(
    batch: Sequence[QuantumScript], precision: int = 20
) -> LiteralString:
    from jaqalpaq.core import Circuit
    from jaqalpaq.core.usepulses import UsePulsesStatement
    from jaqalpaq.generator import generate_jaqal_program
    from jaqalpaq.qsyntax import circuit as jaqal_circuit

    if TYPE_CHECKING:
        from jaqalpaq.qsyntax import qsyntax  # noqa: F401

    # Since jaqal requires a top-level register declaration, we need
    # to find out the required number of qubits for all the tapes
    num_qubits = 0
    for tape in batch:
        sa_res = sa.analyze(tape)
        num_qubits = max(num_qubits, max(sa_res.qubits) + 1)

    @jaqal_circuit(inject_pulses=get_boson_gate_defs(), autoload_pulses="ignore")
    def program(Q: "qsyntax.Q"):
        Q.usepulses("qscout.v1.std")
        q = Q.register(num_qubits, "q")
        for tape in batch:
            with Q.subcircuit(tape.shots.total_shots):
                for op in tape.operations:
                    gate_to_ir(op, Q, q)

    with patch(
        "jaqalpaq.generator.generator._jaqal_value_numeric_context",
        decimal.Context(prec=precision),
    ):
        ir = cast(Circuit, program())  # pyright: ignore[reportCallIssue]
        # Have to defer the usepulses call until generation or it'll throw a ModuleImportError
        ir._usepulses.append(UsePulsesStatement(QUBIT_BOSON_MODULE, all))  # type: ignore[reportPrivateUsage]
        return generate_jaqal_program(ir).strip()


def convert_params(params):
    return [p.item() if hasattr(p, "item") else p for p in params]


@singledispatch
def gate_to_ir(op: Operation, Q, q):
    if gate_id := QUBIT_GATES.get(op.name, None):
        wires = [q[wire] for wire in op.wires]
        getattr(Q, gate_id)(*wires, *convert_params(op.parameters))
        return

    raise DeviceError(f"Cannot serialize non-native gate to Jaqal: {op}")


@gate_to_ir.register
def _(op: native_ops.R, Q, q):
    gate_id = QUBIT_GATES[op.name]
    angle, axis_angle = convert_params(op.parameters)
    qubit = q[op.wires[0]]
    getattr(Q, gate_id)(qubit, axis_angle, angle)


@gate_to_ir.register
def _(_op: qml.GlobalPhase | qml.Identity, Q, q):
    return


@gate_to_ir.register
def _(op: hqml.Red | hqml.Blue, Q, q):
    gate_id = BOSON_GATES[op.name]
    qubit, mode = op.wires
    assert isinstance(mode, Qumode)
    [angle, phase] = convert_params(op.parameters)
    getattr(Q, gate_id)(q[qubit], mode.manifold, mode.index, phase, angle)


@gate_to_ir.register
def _(op: hqml.FockState, Q, q):
    gate_id = BOSON_GATES[op.name]
    fock_state = int(op.hyperparameters["n"])
    qubit, mode = op.wires
    assert isinstance(mode, Qumode)
    getattr(Q, gate_id)(q[qubit], mode.manifold, mode.index, fock_state)


@gate_to_ir.register
def _(op: native_ops.SidebandProbe, Q, q):
    gate_id = BOSON_GATES[op.name]
    [duration_us, phase, sign, detuning] = convert_params(op.parameters)
    qubit, mode = op.wires
    assert isinstance(mode, Qumode)
    getattr(Q, gate_id)(
        q[qubit], phase, duration_us, mode.manifold, mode.index, sign, detuning
    )


@gate_to_ir.register
def _(op: hqml.XCD | hqml.YCD | hqml.CD, Q, q):
    gate_id = BOSON_GATES[op.name]
    qubit, mode = op.wires
    assert isinstance(mode, Qumode)
    [beta, angle] = convert_params(op.parameters)
    beta_re = beta * math.cos(angle)
    beta_im = beta * math.sin(angle)
    getattr(Q, gate_id)(q[qubit], mode.manifold, mode.index, beta_re, beta_im)


@gate_to_ir.register
def _(op: native_ops.ConditionalXSqueezing, Q, q):
    gate_id = BOSON_GATES[op.name]
    qubit, mode = op.wires
    [blue_red_ratio] = convert_params(op.parameters)
    getattr(Q, gate_id)(q[qubit], blue_red_ratio)


@gate_to_ir.register
def _(op: native_ops.NativeBeamsplitter, Q, q):
    gate_id = BOSON_GATES[op.name]
    qubit, *modes = op.wires
    [detuning1, detuning2, duration, phase] = convert_params(op.parameters)
    getattr(Q, gate_id)(q[qubit], detuning1, detuning2, duration, phase)
