# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

import math
from functools import partial, singledispatch
from typing import Any

import pennylane as qp
import pennylane.measurements as pl_mp
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.ops.op_math import (
    CompositeOp,
    Controlled,
    ScalarSymbolicOp,
    SProd,
    SymbolicOp,
)
from pennylane.ops.op_math.condition import Conditional
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn

import hybridlane as hl
from hybridlane.measurements import BasisMap

from .. import measurements as hl_mp
from ..wires import ComputationalBasis, TypeCheckResult, type_check

optional_qumode_measurements = {
    "homodyne": ComputationalBasis.Position,
    "fock": ComputationalBasis.Discrete,
}


@qp.transform
def from_pennylane(
    tape: QuantumScript, default_qumode_measurement: str | None = None
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Transformation that converts pennylane objects to hybridlane ones

    The following transformations are performed on the input program:

    1. Gates in the circuit are mapped to their Hybridlane equivalents if they exist, such as ``qp.Beamsplitter -> hl.Beamsplitter``. Parameters may be transformed if necessary to ensure the original intent of the program is preserved. This transformation also recursively traverses the gate definition for ``SymbolicOps``, like ``qp.adjoint(qp.Beamsplitter) -> qp.adjoint(hl.Beamsplitter)``.

    2. Observables are mapped to Hybridlane equivalents, if necessary, in a similar recursive manner to the gates. This also extends to``CompositeOps``.

    3. Measurement processes are transformed, if possible, as a convenience for users who forget to use the ``hl`` versions, like ``qp.expval -> hl.expval``.

    Args:
        tape: The quantum program to transform

        default_qumode_measurement: A basis to measure qumodes in if a basis could not be inferred (e.g. for qp.sample() passing in wires).
            Must be one of "homodyne" or "fock".
    """

    if (
        default_qumode_measurement
        and default_qumode_measurement not in optional_qumode_measurements
    ):
        raise ValueError(
            f"default_qumode_measurement must be one of {optional_qumode_measurements}, got: {default_qumode_measurement}"
        )

    new_ops = list(map(convert_operator, tape.operations))
    just_ops = QuantumScript(new_ops)

    sa_res = type_check(just_ops)
    cache = {"sa_res": sa_res}
    mp_fn = partial(
        convert_measurement_process,
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


@singledispatch
def convert_operator(op: Operator) -> Operator:
    return op


@convert_operator.register
def _(op: Controlled):
    return qp.ctrl(
        convert_operator(op.base),
        op.control_wires,
        control_values=op.control_values,
        work_wires=op.work_wires,
        work_wire_type=op.work_wire_type,
    )


@convert_operator.register
def _(op: SymbolicOp):
    return op.__class__(convert_operator(op.base))


@convert_operator.register
def _(op: Conditional):
    return op.__class__(op.hyperparameters["meas_val"], convert_operator(op.base))  # ty:ignore[invalid-argument-type]


@convert_operator.register
def _(op: qp.Displacement):
    # no change, just convert to our gate
    return hl.Displacement(*op.data, wires=op.wires)


@convert_operator.register
def _(op: qp.Rotation):
    # i -> -i
    return hl.Rotation(-op.data[0], wires=op.wires)


@convert_operator.register
def _(op: qp.Squeezing):
    # we use re(i2t), they use re(ip), so t = p/2
    r, phi = op.parameters
    return hl.Squeezing(r, phi / 2, wires=op.wires)


@convert_operator.register
def _(op: qp.Kerr):
    # i -> -i
    return hl.Kerr(-op.data[0], wires=op.wires)


@convert_operator.register
def _(op: qp.CubicPhase):
    # ir(x^3)/3 -> -irx^3
    return hl.CubicPhase(-op.data[0] / 3, wires=op.wires)


@convert_operator.register
def _(op: qp.Beamsplitter):
    # θ(e^{iϕ} ab† - e^{-iϕ} a†b) -> -iθ'/2 (e^{iϕ'} a†b + e^{iϕ'} ab†)
    # θ' = 2θ
    # ϕ' = -(ϕ + π/2)
    theta, phi = op.data
    return hl.Beamsplitter(2 * theta, -(phi + math.pi / 2), wires=op.wires)


@convert_operator.register
def _(op: qp.TwoModeSqueezing):
    # r -> -r
    r, phi = op.data
    return hl.TwoModeSqueezing(-r, phi, wires=op.wires)


@singledispatch
def convert_observable(obs: Operator) -> Operator:
    return obs


@convert_observable.register
def _(obs: SProd) -> SProd:
    return obs.__class__(obs.scalar, convert_observable(obs.base))


@convert_observable.register
def _(obs: ScalarSymbolicOp) -> ScalarSymbolicOp:
    return obs.__class__(convert_observable(obs.base), obs.scalar)


@convert_observable.register
def _(obs: SymbolicOp) -> SymbolicOp:
    return obs.__class__(convert_observable(obs.base))


@convert_observable.register
def _(obs: CompositeOp) -> CompositeOp:
    operands = [convert_observable(op) for op in obs.operands]
    return obs.__class__(*operands)


@convert_observable.register
def _(obs: qp.QuadX):
    return hl.QuadX(wires=obs.wires)


@convert_observable.register
def _(obs: qp.QuadP):
    return hl.QuadP(wires=obs.wires)


@convert_observable.register
def _(obs: qp.QuadOperator):
    return hl.QuadOperator(*obs.data, wires=obs.wires)


@convert_observable.register
def _(obs: qp.FockStateProjector):
    return hl.FockStateProjector(*obs.data, wires=obs.wires)


@singledispatch
def convert_measurement_process(
    mp: MeasurementProcess,
    *,
    default_qumode_measurement: str | None = None,
    cache: dict[str, Any] | None = None,
) -> MeasurementProcess:
    return mp


# We stack decorators to handle the OR `|` cases from the match statement.
@convert_measurement_process.register(pl_mp.ExpectationMP)
@convert_measurement_process.register(hl_mp.ExpectationMP)
def _(
    mp: pl_mp.ExpectationMP | hl_mp.ExpectationMP,
    *,
    default_qumode_measurement: str | None = None,
    cache: dict[str, Any] | None = None,
) -> hl_mp.ExpectationMP:
    if mp.obs:
        return hl_mp.ExpectationMP(obs=convert_observable(mp.obs))
    raise NotImplementedError("An observable is required with hl.expval")


@convert_measurement_process.register(pl_mp.VarianceMP)
@convert_measurement_process.register(hl_mp.VarianceMP)
def _(
    mp: pl_mp.VarianceMP | hl_mp.VarianceMP,
    *,
    default_qumode_measurement: str | None = None,
    cache: dict[str, Any] | None = None,
):
    if mp.obs:
        return hl_mp.VarianceMP(obs=convert_observable(mp.obs))
    raise NotImplementedError("An observable is required with hl.var")


@convert_measurement_process.register
def _(
    mp: hl_mp.SampleMP,
    *,
    default_qumode_measurement: str | None = None,
    cache: dict[str, Any] | None = None,
):
    if mp.obs:
        return hl_mp.SampleMP(obs=convert_observable(mp.obs))
    return mp


@convert_measurement_process.register
def _(
    mp: pl_mp.SampleMP,
    *,
    default_qumode_measurement: str | None = None,
    cache: dict[str, Any] | None = None,
):
    if mp.obs:
        return hl_mp.SampleMP(obs=convert_observable(mp.obs))

    sa_res: TypeCheckResult = cache["sa_res"]
    schema = BasisMap(
        {q: ComputationalBasis.Discrete for q in mp.wires & sa_res.qubits}
    )
    if sa_res.qumodes:
        if default_qumode_measurement is None:
            raise ValueError(
                f"Unable to infer basis measurements for qumodes {sa_res.qumodes}. "
                "Consider passing in the `default_qumode_measurement` argument"
            )
        fill_value = optional_qumode_measurements[default_qumode_measurement]
        qumode_schema = BasisMap({m: fill_value for m in mp.wires & sa_res.qumodes})
        schema |= qumode_schema

    return hl_mp.SampleMP(bases=schema)
