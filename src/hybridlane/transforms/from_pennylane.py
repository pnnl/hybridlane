# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import math
from functools import partial
from typing import Any, Optional

import pennylane as qml
import pennylane.measurements as pl_mp
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.ops.op_math import CompositeOp, ScalarSymbolicOp, SProd, SymbolicOp
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires

import hybridlane as hqml

from .. import measurements as hl_mp
from .. import sa

optional_qumode_measurements = {
    "homodyne": sa.ComputationalBasis.Position,
    "fock": sa.ComputationalBasis.Discrete,
}


@qml.transform
def from_pennylane(
    tape: QuantumScript, default_qumode_measurement: Optional[str] = None
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Transformation that converts pennylane objects to hybridlane ones

    The following transformations are performed on the input program:

    1. Gates in the circuit are mapped to their Hybridlane equivalents if they exist, such as ``qml.Beamsplitter -> hqml.Beamsplitter``. Parameters may be transformed if necessary to ensure the original intent of the program is preserved. This transformation also recursively traverses the gate definition for ``SymbolicOps``, like ``qml.adjoint(qml.Beamsplitter) -> qml.adjoint(hqml.Beamsplitter)``.

    2. Observables are mapped to Hybridlane equivalents, if necessary, in a similar recursive manner to the gates. This also extends to``CompositeOps``.

    3. Measurement processes are transformed, if possible, as a convenience for users who forget to use the ``hqml`` versions, like ``qml.expval -> hqml.expval``.

    Args:
        tape: The quantum program to transform

        default_qumode_measurement: A basis to measure qumodes in if a basis could not be inferred (e.g. for qml.sample() passing in wires).
            Must be one of "homodyne" or "fock".
    """

    if (
        default_qumode_measurement
        and default_qumode_measurement not in optional_qumode_measurements
    ):
        raise ValueError(
            f"default_qumode_measurement must be one of {optional_qumode_measurements}, got: {default_qumode_measurement}"
        )

    new_ops = list(map(_convert_operator, tape.operations))
    just_ops = QuantumScript(new_ops)

    sa_res = sa.analyze(just_ops)
    cache = {"sa_res": sa_res}
    mp_fn = partial(
        _convert_measurement_process,
        default_qumode_measurement=default_qumode_measurement,
        cache=cache,
    )
    new_mps = list(map(mp_fn, tape.measurements))
    new_tape = QuantumScript(
        new_ops, new_mps, shots=tape.shots, trainable_params=tape.trainable_params
    )

    def null_postprocessing(results):
        return results[0]

    return [new_tape], null_postprocessing


def _convert_operator(op: Operator) -> Operator:
    # Handle qml.adjoint
    if isinstance(op, SymbolicOp):
        return op.__class__(_convert_observable(op.base), id=op.id)

    match op:
        # No change
        case qml.Displacement(data=data, wires=wires, id=id):
            return hqml.Displacement(*data, wires=wires, id=id)

        # i -> -i
        case qml.Rotation(data=data, wires=wires, id=id):
            return hqml.Rotation(-data[0], wires=wires, id=id)

        # No change
        case qml.Squeezing(data=data, wires=wires, id=id):
            return hqml.Squeezing(*data, wires=wires, id=id)

        # i -> -i
        case qml.Kerr(data=data, wires=wires, id=id):
            return hqml.Kerr(-data[0], wires=wires, id=id)

        # ir(x^3)/3 -> -irx^3
        case qml.CubicPhase(data=data, wires=wires, id=id):
            return hqml.CubicPhase(-data[0] / 3, wires=wires, id=id)

        # θ(e^{iϕ} ab† - e^{-iϕ} a†b) -> -iθ'/2 (e^{iϕ'} a†b + e^{iϕ'} ab†)
        # θ' = 2θ
        # ϕ' = -(ϕ + π/2)
        case qml.Beamsplitter(data=data, wires=wires, id=id):
            theta, phi = data
            return hqml.Beamsplitter(
                2 * theta, -(phi + math.pi / 2), wires=wires, id=id
            )

        # r -> -r
        case qml.TwoModeSqueezing(data=data, wires=wires, id=id):
            r, phi = data
            return hqml.TwoModeSqueezing(-r, phi, wires=wires, id=id)

        case _:
            return op


def _convert_observable(obs: Operator) -> Operator:
    # Traverse the observable tree, if required.
    # For some reason, SProd breaks the convention
    if isinstance(obs, SProd):
        return obs.__class__(obs.scalar, _convert_observable(obs.base), id=obs.id)
    if isinstance(obs, ScalarSymbolicOp):
        return obs.__class__(_convert_observable(obs.base), obs.scalar, id=obs.id)
    if isinstance(obs, SymbolicOp):
        return obs.__class__(_convert_observable(obs.base), id=obs.id)
    elif isinstance(obs, CompositeOp):
        operands = list(map(lambda op: _convert_observable(op), obs.operands))
        return obs.__class__(*operands, id=id)

    match obs:
        case qml.QuadX(wires=wires):
            return hqml.QuadX(wires=wires)

        case qml.QuadP(wires=wires):
            return hqml.QuadP(wires=wires)

        case qml.QuadOperator(data=data, wires=wires):
            return hqml.QuadOperator(*data, wires=wires)

        case qml.FockStateProjector(data=data, wires=wires):
            return hqml.FockStateProjector(*data, wires=wires)

        case _:
            return obs


def _convert_measurement_process(
    mp: MeasurementProcess,
    default_qumode_measurement: Optional[str] = None,
    cache: Optional[dict[str, Any]] = None,
) -> MeasurementProcess:
    match mp:
        case pl_mp.ExpectationMP(obs=obs, id=id) | hl_mp.ExpectationMP(obs=obs, id=id):
            if obs:
                return hl_mp.ExpectationMP(obs=_convert_observable(obs), id=id)

            raise NotImplementedError("An observable is required with hqml.expval")

        case pl_mp.VarianceMP(obs=obs, id=id) | hl_mp.VarianceMP(obs=obs, id=id):
            if obs:
                return hl_mp.VarianceMP(obs=_convert_observable(obs), id=id)

            raise NotImplementedError("An observable is required with hqml.var")

        case hl_mp.SampleMP(obs=obs, schema=schema, id=id):
            if obs:
                return hl_mp.SampleMP(obs=_convert_observable(obs), id=id)

            return mp

        case pl_mp.SampleMP(obs=obs, wires=wires, id=id):
            if obs:
                return hl_mp.SampleMP(obs=_convert_observable(obs), id=id)

            # If we have no observable, we need to find which wires are qubits and qumodes,
            # then make all qumodes use ``default_qumode_measurement``, or error if its not provided
            # since we have no way of inferring which basis to use.
            sa_res: sa.StaticAnalysisResult = cache["sa_res"]
            schema = sa.BasisSchema(
                {
                    Wires(q): sa.ComputationalBasis.Discrete
                    for q in wires & sa_res.qubits
                }
            )
            if sa_res.qumodes:
                if default_qumode_measurement is None:
                    raise ValueError(
                        f"Unable to infer basis measurements for qumodes {sa_res.qumodes}. "
                        "Consider passing in the `default_qumode_measurement` argument"
                    )

                fill_value = optional_qumode_measurements[default_qumode_measurement]
                qumode_schema = sa.BasisSchema(
                    {Wires(m): fill_value for m in wires & sa_res.qumodes}
                )
                schema |= qumode_schema

            return hl_mp.SampleMP(schema=schema, id=id)

        case _:
            return mp
