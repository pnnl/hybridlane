# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import functools
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Hashable, cast

import pennylane as qp
from pennylane.measurements import MeasurementProcess
from pennylane.operation import CVObservable, CVOperation, Operator
from pennylane.ops import (
    CompositeOp,
    Controlled,
    ControlledOp,
    SymbolicOp,
)
from pennylane.tape import QuantumScript
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from ..ops import QubitConditioned
from ..ops.mixins import Hybrid, Spectral
from .base import (
    BasisMap,
    ComputationalBasis,
    Qubit,
    Qudit,
    Qumode,
    TypeCheckResult,
    TypedWire,
    WireType,
)
from .exceptions import TypeCheckError

Context = dict[WiresLike, WireType]


@functools.singledispatch
@functools.lru_cache(maxsize=128)
def type_check(tape: QuantumScript) -> TypeCheckResult:
    """Core hybridlane type-checking routine

    This performs type inference over the circuit wires based on operations and measurements
    in the circuit.

    **Example**

    It can be applied to a QNode similar to ``qp.specs``:

    .. code-block:: python

        dev = qp.device("default.hybrid", fock_level=8)

        @qp.qnode(dev)
        def circuit(alpha):
            qp.CatState(alpha, 0, 0, wires=0)
            hl.D(alpha, 0, 0)
            qp.H(1)
            hl.SQR(np.pi, np.pi / 2, 0, wires=[1, 0])
            qp.H(1)
            return hl.expval(qp.Z(1))

    >>> res = hl.type_check(circuit)(2.0)
    >>> print(res.wire_types)
    OrderedDict({0: Qumode(), 1: Qubit()})

    It can also be directly applied to a tape:

    >>> qs = qp.tape.QuantumScript([qp.X(0), hl.D(0.5, 0, wires=1)])
    >>> res = hl.type_check(qs)
    >>> print(res.wire_types)
    OrderedDict({0: Qubit(), 1: Qumode()})

    **Details**

    The type checking procedure performs a few checks over the circuit:

    1. Wire types are determined by inspecting each operation in the circuit. If there's a
    conflict, an error is raised.

    2. It tries to determine the type of a wire from the measurement performed on it, usually
    based on the observable. Any conflicts with types determined from the operations will
    also raise an error.

    3. We also try to determine the type of measurement required (fock readout, homodyne),
    particularly for sample-based measurements.

    Certain operations carry no type information, such as ``qp.Identity`` or ``qp.Snapshot``.
    hybridlane supports static typing through functionality similar to
    :py:func:`pennylane.registers`, with the functions :py:func:`~hybridlane.qubits` and
    :py:func:`~hybridlane.qumodes`.

    Raises:
        :py:class:`~hybridlane.wires.TypeCheckError`: if there's any error in analyzing the
            circuit structure
    """
    # Read statically typed wires
    context = {}
    for wire in tape.wires:
        if isinstance(wire, TypedWire):
            context[wire] = wire.wire_type

    context = infer_wires(tape.operations, context)

    measurement_schemas: list[BasisMap | None] = []
    if tape.measurements:
        for m in tape.measurements:
            basis_map = infer_measurement_bases(m, context)
            measurement_schemas.append(basis_map)
            m_wires = infer_wires(m, context)

            if before_after := _validate_context_diff(context, m_wires):
                raise TypeCheckError(_helpful_error_message(m, before_after))

            context |= m_wires

    if missing_wires := tape.wires - context.keys():
        raise TypeCheckError(f"Unable to infer wire types for {missing_wires}")

    # Order the wire types to match the tape wire order
    ordered_wire_types = OrderedDict()
    for w in tape.wires:
        ordered_wire_types[w] = context[w]

    return TypeCheckResult(ordered_wire_types, measurement_schemas)


@type_check.register
def _type_check_qnode(
    qnode: qp.QNode,
) -> Callable[..., TypeCheckResult | list[TypeCheckResult]]:
    def wrapper(*args, **kwargs) -> TypeCheckResult | list[TypeCheckResult]:
        batch, _ = qp.workflow.construct_batch(qnode)(*args, **kwargs)
        results = [type_check(tape) for tape in batch]

        if len(results) == 1:
            return results[0]

        return results

    return wrapper


@functools.singledispatch
def infer_wires(obj, context: Context) -> Context:
    if isinstance(obj, Sequence):
        return _infer_wires_from_ops(obj, context)

    raise TypeCheckError(f"Unknown how to infer type of {obj}")


def _infer_wires_from_ops(ops: Sequence[Operator], context) -> Context:
    for op in ops:
        new_context = infer_wires(op, context)

        if before_after := _validate_context_diff(context, new_context):
            raise TypeCheckError(_helpful_error_message(op, before_after))

        context = new_context

    return context


@infer_wires.register
def _infer_wire_types_from_operator(op: Operator, context: Context) -> Context:
    # todo: rework for graph decomposition system
    if op.has_decomposition:
        return infer_wires(op.decomposition(), context)

    module = op.__class__.__module__
    if module.startswith("pennylane.ops.qubit"):
        return context | {w: Qubit() for w in op.wires}
    elif module.startswith("pennylane.ops.qutrit"):
        return context | {w: Qudit(3) for w in op.wires}

    raise TypeCheckError(f"Unable to infer wire types for operation {op}")


@infer_wires.register
def _(
    _: qp.Identity | qp.Snapshot | qp.GlobalPhase | qp.WireCut | qp.Barrier,
    context: Context,
):
    # None of these gates add any constraints on wire types
    return context


@infer_wires.register
def _(op: CVOperation | CVObservable, context: Context):
    return context | {w: Qumode() for w in op.wires}


@infer_wires.register
def _(op: Hybrid, context: Context):
    return context | op.wire_types()


@infer_wires.register
def _(op: SymbolicOp, context: Context):
    return infer_wires(op.base, context)


@infer_wires.register
def _(op: CompositeOp, context: Context):
    return infer_wires(op.operands, context)


@infer_wires.register
def _(op: Controlled | ControlledOp | QubitConditioned, context: Context):
    control_wires = {w: Qubit() for w in op.control_wires}
    return context | control_wires | infer_wires(op.base, context)


@infer_wires.register
def _(
    op: qp.BasisState | qp.StatePrep | qp.Superposition,
    context: Context,
):
    return context | {w: Qubit() for w in op.wires}


@infer_wires.register
def _infer_from_measurement(m: MeasurementProcess, context) -> Context:
    from ..measurements import (
        SampleMeasurement,
        StateMeasurement,
    )

    if m.obs is not None:
        return infer_wires(m.obs, context)

    elif isinstance(m, StateMeasurement):
        return context

    elif isinstance(m, SampleMeasurement):
        return infer_wires(m.schema, context)

    return context


@infer_wires.register
def _infer_wire_types_from_basis_map(map: BasisMap, context) -> Context:
    new_context = {}

    for wire in map.wires:
        match map[wire]:
            case ComputationalBasis.Position | ComputationalBasis.Coherent:
                new_context[wire] = Qumode()
            case ComputationalBasis.Discrete:
                # Not enough information to infer, since DV measurements could be qubit or Fock
                pass

    return context | new_context


# todo: maybe incorporate the attributes.diagonal_in_fock_basis and attributes.diagonal_in_position_basis?
@functools.singledispatch
def infer_measurement_bases(obs: Operator, context) -> BasisMap:
    # Qubit observables are automatically discrete
    if obs.pauli_rep is not None:
        return BasisMap({obs.wires: ComputationalBasis.Discrete})

    raise TypeCheckError(f"No known way to infer decomposition for observable {obs}")


@infer_measurement_bases.register
def _(obs: CompositeOp, context) -> BasisMap:
    return BasisMap.all_wires(
        [infer_measurement_bases(o, context) for o in obs.operands]
    )


@infer_measurement_bases.register
def _(obs: SymbolicOp, context) -> BasisMap:
    return infer_measurement_bases(obs.base, context)


@infer_measurement_bases.register
def _(obs: Spectral, context) -> BasisMap:
    return BasisMap({cast(Operator, obs).wires: obs.natural_basis})


@infer_measurement_bases.register
def _(mp: MeasurementProcess, context) -> BasisMap:
    from ..measurements import SampleMeasurement

    if mp.obs:
        return infer_measurement_bases(mp.obs, context)

    if isinstance(mp, SampleMeasurement):
        return mp.schema

    # State measurements with no observables reach here
    return BasisMap({})


def infer_bases_from_tensors(tensors: dict[Hashable, TensorLike]) -> BasisMap:
    r"""Constructs a schema from the provided tensors using their data types

    Args:
        tensors: A mapping from wires to tensors

    Raises:
        :py:class:`ValueError`: if any of the tensors don't have an ``int``, ``float``, or ``complex`` like datatype
    """
    wire_map = {}
    for wire, tensor in tensors.items():
        dtype: str = qp.math.get_dtype_name(tensor)

        if dtype.startswith("int") or dtype.startswith("uint"):
            basis = ComputationalBasis.Discrete
        elif dtype.startswith("float"):
            basis = ComputationalBasis.Position
        elif dtype.startswith("complex"):
            basis = ComputationalBasis.Coherent
        else:
            raise TypeCheckError(f"Unrecognized dtype: {dtype}")

        wire_map[wire] = basis

    return BasisMap(wire_map)


def _validate_context_diff(
    wire_types: Context, new_wire_types: Context
) -> dict[WiresLike, tuple[WireType, WireType]]:
    before_after = {}
    aliased_wires = new_wire_types.keys() & wire_types.keys()
    for wire in aliased_wires:
        # For repeated wire usage e.g. X(0) Z(0), it could be correct. We have to iterate to
        # see if any wires are different from previously decided types.
        if wire_types[wire] != new_wire_types[wire]:
            before_after[wire] = (wire_types[wire], new_wire_types[wire])

    return before_after


def _helpful_error_message(
    obj: Operator | MeasurementProcess,
    before_after: dict[WiresLike, tuple[WireType, WireType]],
) -> str:

    match obj:
        case Operator():
            msg = f"Operation {obj} is incompatible with previous circuit operations.\n"
        case MeasurementProcess():
            msg = f"Measurement {obj} is incompatible with previous circuit operations or measurements.\n"

    for wire, (before, after) in before_after.items():
        msg += f" - Wire {wire} was previously inferred as {before}, but is now inferred as {after}"

    return msg
