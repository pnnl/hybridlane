# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import functools
from collections import OrderedDict
from typing import Hashable

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import CV, Operator
from pennylane.ops import CompositeOp, Controlled, ControlledOp, SymbolicOp
from pennylane.tape import QuantumScript
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from ..measurements import (
    SampleMeasurement,
    StateMeasurement,
)
from ..ops import QubitConditioned
from ..ops.mixins import Hybrid, Spectral
from .base import (
    BasisSchema,
    ComputationalBasis,
    Qubit,
    Qumode,
    StaticAnalysisResult,
    WireType,
)
from .exceptions import StaticAnalysisError


@functools.lru_cache(maxsize=128)
def analyze(
    tape: QuantumScript, fill_missing: WireType | None = None
) -> StaticAnalysisResult:
    """Static circuit analysis pass to identify wire types and measurement schemas

    This function performs a number of checks:

    1. It infers the type of a wire (qubit/qumode) from the operations that act on it.

    2. If that fails, it tries to determine the type of a wire from the measurement performed on it, usually
    based on the observable.

    3. We also try to determine the type of measurement required (fock readout, homodyne), particularly
    for sample-based measurements.

    If it finds that a wire is used as both a qubit and a qumode, it will raise an error.

    Args:
        tape: The quantum circuit to analyze

        fill_missing: An optional wire type specifying what default to provide for unidentified wires

    Raises:
        :py:class:`~hybridlane.sa.exceptions.StaticAnalysisError` if there's any error in analyzing the circuit
        structure
    """
    # The strategy:
    #  1. Wire types can be determined by operations.
    #  2. Different measurement processes may have different schemas, and this tells us
    #     about wire types as well. They must agree with wire types inferred from operations.
    #  3. There may be wires we can't find a type for (which means it likely doesn't even participate
    #     in the circuit in a meaningful way.. why was it even defined?)

    wire_types = _infer_wire_types_from_operators(tape.operations)

    measurement_schemas: list[BasisSchema | None] = []
    if tape.measurements:
        for m in tape.measurements:
            schema = infer_schema_from_measurement(m)
            measurement_schemas.append(schema)
            m_wires = _infer_wire_types_from_measurement(m)

            if before_after := _validate_aliased_wires(wire_types, m_wires):
                raise StaticAnalysisError(_aliased_wire_msg_helper(before_after, m=m))

            wire_types |= m_wires

    if missing_wires := tape.wires - wire_types.keys():
        if fill_missing is not None:
            wire_types |= {w: fill_missing for w in missing_wires}
        else:
            raise StaticAnalysisError(f"Unable to infer wire types for {missing_wires}")

    # Order the wire types to match the tape wire order
    ordered_wire_types = OrderedDict()
    for w in tape.wires:
        ordered_wire_types[w] = wire_types[w]

    return StaticAnalysisResult(ordered_wire_types, measurement_schemas)


def _infer_wire_types_from_operators(ops: list[Operator]) -> dict[WiresLike, WireType]:
    wire_types: dict[WiresLike, WireType] = {}

    for op in ops:
        new_wire_types = _infer_wire_types_from_operator(op)

        if before_after := _validate_aliased_wires(wire_types, new_wire_types):
            raise StaticAnalysisError(_aliased_wire_msg_helper(before_after, op=op))

        wire_types |= new_wire_types

    return wire_types


@functools.singledispatch
def _infer_wire_types_from_operator(op: Operator) -> dict[WiresLike, WireType]:
    if op.has_decomposition:
        return _infer_wire_types_from_operators(op.decomposition())

    return {w: Qubit() for w in op.wires}


@_infer_wire_types_from_operator.register
def _(op: CV):
    return {w: Qumode() for w in op.wires}


@_infer_wire_types_from_operator.register
def _(op: Hybrid):
    return op.wire_types()


@_infer_wire_types_from_operator.register
def _(op: SymbolicOp):
    return _infer_wire_types_from_operator(op.base)


@_infer_wire_types_from_operator.register
def _(op: CompositeOp):
    return _infer_wire_types_from_operators(op.operands)


@_infer_wire_types_from_operator.register
def _(op: Controlled | ControlledOp | QubitConditioned):
    wire_types = {w: Qubit() for w in op.control_wires}
    wire_types |= _infer_wire_types_from_operator(op.base)
    return wire_types


def _infer_wire_types_from_measurement(
    m: MeasurementProcess,
) -> dict[WiresLike, WireType]:
    if m.obs is not None:
        return _infer_wire_types_from_observable(m.obs)

    # Fixme: State measurements with no observable don't have enough information, we'd have
    # to obtain the truncation too
    elif isinstance(m, StateMeasurement):
        return {}

    elif isinstance(m, SampleMeasurement):
        return _infer_wire_types_from_schema(m.schema)

    return {}


def _infer_wire_types_from_schema(schema: BasisSchema) -> dict[WiresLike, WireType]:
    wire_types = {}

    for wire in schema.wires:
        match schema.get_basis(wire):
            case ComputationalBasis.Position | ComputationalBasis.Coherent:
                wire_types[wire] = Qumode()
            case ComputationalBasis.Discrete:
                # Not enough information to infer, since DV measurements could be qubit or Fock
                pass

    return wire_types


@functools.singledispatch
def _infer_wire_types_from_observable(obs: Operator) -> dict[WiresLike, WireType]:
    return _infer_wire_types_from_operator(obs)


@_infer_wire_types_from_observable.register
def _(obs: CompositeOp):
    # Override to have a custom message
    wire_types = {}
    for op in obs.operands:
        new_wire_types = _infer_wire_types_from_operator(op)

        if before_after := _validate_aliased_wires(wire_types, new_wire_types):
            raise StaticAnalysisError(
                f"Observable {obs} treats wires {list(before_after.keys())} as multiple types"
            )

        wire_types |= new_wire_types

    return wire_types


# todo: maybe incorporate the attributes.diagonal_in_fock_basis and attributes.diagonal_in_position_basis?
def infer_schema_from_observable(obs: Operator) -> BasisSchema:
    if isinstance(obs, CompositeOp):
        return BasisSchema.all_wires(
            [infer_schema_from_observable(o) for o in obs.operands]
        )

    # Scalar doesn't change the schema, and O^d can be diagonalized all the same
    elif isinstance(obs, SymbolicOp):
        return infer_schema_from_observable(obs.base)

    # CV operators that we've given a spectrum can be inferred
    elif isinstance(obs, Spectral):
        return BasisSchema({obs.wires: obs.natural_basis})

    # Qubit observables are automatically discrete
    elif obs.pauli_rep is not None:
        return BasisSchema({obs.wires: ComputationalBasis.Discrete})

    raise StaticAnalysisError(
        f"No known way to infer decomposition for observable {obs}"
    )


def infer_schema_from_measurement(m: MeasurementProcess) -> BasisSchema | None:
    if m.obs:
        return infer_schema_from_observable(m.obs)

    if isinstance(m, SampleMeasurement):
        return m.schema

    # State measurements with no observables reach here
    return None


def infer_schema_from_tensors(tensors: dict[Hashable, TensorLike]) -> BasisSchema:
    r"""Constructs a schema from the provided tensors using their data types

    Args:
        tensors: A mapping from wires to tensors

    Raises:
        :py:class:`ValueError`: if any of the tensors don't have an ``int``, ``float``, or ``complex`` like datatype
    """
    wire_map = {}
    for wire, tensor in tensors.items():
        dtype: str = qml.math.get_dtype_name(tensor)

        if dtype.startswith("int") or dtype.startswith("uint"):
            basis = ComputationalBasis.Discrete
        elif dtype.startswith("float"):
            basis = ComputationalBasis.Position
        elif dtype.startswith("complex"):
            basis = ComputationalBasis.Coherent
        else:
            raise StaticAnalysisError(f"Unrecognized dtype: {dtype}")

        wire_map[wire] = basis

    return BasisSchema(wire_map)


def _validate_aliased_wires(
    wire_types: dict[WiresLike, WireType], new_wire_types: dict[WiresLike, WireType]
) -> dict[WiresLike, tuple[WireType, WireType]]:
    before_after = {}
    aliased_wires = new_wire_types.keys() & wire_types.keys()
    for wire in aliased_wires:
        # For repeated wire usage e.g. X(0) Z(0), it could be correct. We have to iterate to
        # see if any wires are different from previously decided types.
        if wire_types[wire] != new_wire_types[wire]:
            before_after[wire] = (wire_types[wire], new_wire_types[wire])

    return before_after


def _aliased_wire_msg_helper(
    before_after: dict[WiresLike, tuple[WireType, WireType]],
    op: Operator | None = None,
    m: MeasurementProcess | None = None,
) -> str:
    if m:
        msg = f"Measurement {m} is incompatible with previous circuit operations or measurements.\n"
    elif op:
        msg = f"Operation {op} is incompatible with previous circuit operations.\n"

    for wire, (before, after) in before_after.items():
        msg += f" - Wire {wire} was previously inferred as {before}, but is now inferred as {after}"

    return msg
