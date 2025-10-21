# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import functools
from typing import Hashable

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import CV, Operator
from pennylane.ops import CompositeOp, ControlledOp, SymbolicOp
from pennylane.tape import QuantumScript
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ..measurements import (
    SampleMeasurement,
    StateMeasurement,
)
from ..ops.mixins import Hybrid, Spectral
from .base import BasisSchema, ComputationalBasis, StaticAnalysisResult
from .exceptions import StaticAnalysisError


@functools.lru_cache(maxsize=128)
def analyze(
    tape: QuantumScript, fill_missing: str | None = None
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

        fill_missing: An optional string of ``("qubits", "qumodes")`` specifying what default
            to provide for unidentified wires

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

    valid_options = ("qubit", "qumode", None)
    if fill_missing not in valid_options:
        raise ValueError(
            f"Unrecognized fill option {fill_missing}, must be one of {valid_options}"
        )

    qumodes, qubits = _infer_wire_types_from_operations(tape.operations)

    if common_wires := qumodes & qubits:
        raise StaticAnalysisError(
            f"Wires {common_wires} were determined to be both qumodes and qubits from the operations"
        )

    measurement_schemas: list[BasisSchema | None] = []
    if tape.measurements:
        for m in tape.measurements:
            schema = infer_schema_from_measurement(m)
            measurement_schemas.append(schema)

            m_qumodes, m_qubits = _infer_wire_types_from_measurement(m)

            if common_wires := m_qumodes & m_qubits:
                raise StaticAnalysisError(
                    f"Measurement treats wires {common_wires} as both qubit and qumode"
                )

            were_qumodes = qumodes & m_qubits
            were_qubits = qubits & m_qumodes
            if were_qumodes or were_qubits:
                raise StaticAnalysisError(
                    _aliased_wire_msg_helper(were_qumodes, were_qubits, m)
                )

            qumodes += m_qumodes
            qubits += m_qubits

    if missing_wires := tape.wires - (qumodes + qubits):
        if fill_missing is not None:
            if fill_missing == "qubit":
                qubits += missing_wires
            elif fill_missing == "qumode":
                qumodes += missing_wires
        else:
            raise StaticAnalysisError(f"Unable to infer wire types for {missing_wires}")

    return StaticAnalysisResult(qumodes, qubits, measurement_schemas, tape.wires)


def _aliased_wire_msg_helper(
    were_qumodes: Wires, were_qubits: Wires, m: MeasurementProcess
) -> str:
    msg = f"Measurement {m} is incompatible with previous circuit operations or measurements: "

    if were_qumodes:
        msg += f"The wires {were_qumodes} were previously inferred to be qumodes, but are now treated as qubits."

    if were_qubits:
        if were_qumodes:
            msg += " "

        msg += f"The wires {were_qubits} were previously inferred to be qubits, but are now treated as qumodes."

    return msg


def infer_wire_types(
    tape: QuantumScript, fill_missing: str | None = None
) -> tuple[Wires, Wires]:
    """Statically analyzes a tape to partition wires into qumodes and qubits

    This function doesn't provide any validation for some possible errors that might arise
    from a poorly specified tape:

    1. A wire may appear in both ``qumodes`` and ``qubits`` if the tape performed any aliasing.

    2. If the type of a wire cannot be inferred from the tape - because no operation is defined on it, or
    the measurements don't provide enough information - it will be missing from both ``qumodes`` and ``qubits``.
    Missing wires can optionally be given a type through the ``fill_missing`` keyword.

    Args:
        tape: The quantum circuit to analyze

        fill_missing: An optional string of ``("qubits", "qumodes")`` specifying what default
            to provide for unidentified wires
    """

    qumodes, qubits = _infer_wire_types_from_operations(tape.operations)

    if tape.measurements:
        measurement_sets = list(
            map(_infer_wire_types_from_measurement, tape.measurements)
        )
        qumode_sets, qubit_sets = list(zip(*measurement_sets))
        qumodes += Wires.all_wires(qumode_sets)
        qubits += Wires.all_wires(qubit_sets)

    if fill_missing is not None:
        valid_options = ("qubit", "qumode")
        if fill_missing not in valid_options:
            raise ValueError(
                f"Unrecognized fill option {fill_missing}, must be one of {valid_options}"
            )

        missing_wires = tape.wires - (qumodes + qubits)
        if fill_missing == "qubit":
            qubits += missing_wires
        elif fill_missing == "qumode":
            qumodes += missing_wires

    return qumodes, qubits


def _infer_wire_types_from_operations(ops: list[Operator]) -> tuple[Wires, Wires]:
    qumodes, qubits = Wires([]), Wires([])

    for op in ops:
        new_qumodes, new_qubits = _infer_wires_from_operation(op)
        qumodes += new_qumodes
        qubits += new_qubits

    return qumodes, qubits


@functools.singledispatch
def _infer_wires_from_operation(op: Operator):
    qumodes, qubits = Wires([]), Wires([])

    if op.has_decomposition:
        for o in op.decomposition():
            new_qumodes, new_qubits = _infer_wires_from_operation(o)
            qumodes += new_qumodes
            qubits += new_qubits

    else:
        qubits = op.wires

    return qumodes, qubits


@_infer_wires_from_operation.register
def _(op: CV):
    return op.wires, Wires([])


@_infer_wires_from_operation.register
def _(op: Hybrid):
    qubits, qumodes = op.split_wires()
    return qumodes, qubits


@_infer_wires_from_operation.register
def _(op: SymbolicOp):
    return _infer_wires_from_operation(op.base)


@_infer_wires_from_operation.register
def _(op: ControlledOp):
    ctrl_qubits = op.control_wires
    qumodes, qubits = _infer_wires_from_operation(op.base)
    return qumodes, qubits + ctrl_qubits


def _infer_wire_types_from_measurement(
    m: MeasurementProcess,
) -> tuple[Wires, Wires]:
    if m.obs is not None:
        return _infer_wire_types_from_observable(m.obs)

    # Fixme: State measurements with no observable don't have enough information, we'd have
    # to obtain the truncation too
    elif isinstance(m, StateMeasurement):
        return Wires([]), Wires([])

    elif isinstance(m, SampleMeasurement):
        return _infer_wire_types_from_schema(m.schema)

    # Fall through
    return Wires([]), Wires([])


def _infer_wire_types_from_schema(schema: BasisSchema) -> tuple[Wires, Wires]:
    qumodes, qubits = Wires([]), Wires([])

    for wire in schema.wires:
        match schema.get_basis(wire):
            case ComputationalBasis.Position | ComputationalBasis.Coherent:
                qumodes += wire
            case ComputationalBasis.Discrete:
                # Not enough information to infer, since DV measurements could be qubit or Fock
                pass

    return qumodes, qubits


def _infer_wire_types_from_observable(obs: Operator) -> tuple[Wires, Wires]:
    if isinstance(obs, CompositeOp):
        wire_types = list(map(_infer_wire_types_from_observable, obs.operands))
        qumode_sets, qubit_sets = list(zip(*wire_types))
        return Wires.all_wires(qumode_sets), Wires.all_wires(qubit_sets)

    elif isinstance(obs, SymbolicOp):
        return _infer_wire_types_from_observable(obs.base)

    # Specifically doesn't tell us anything since it can apply to any quantum object
    elif isinstance(obs, qml.Identity):
        return Wires([]), Wires([])

    elif obs.pauli_rep:
        return Wires([]), obs.wires  # qubit

    elif isinstance(obs, CV):
        return obs.wires, Wires([])  # qumode

    else:
        raise StaticAnalysisError(f"Unknown how to infer qumodes for observable {obs}")


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

    # raise StaticAnalysisError(
    #     f"Unable to determine schema for measurement {m} because it doesn't "
    #     "have an observable or schema"
    # )


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

        wire_map[Wires(wire)] = basis

    return BasisSchema(wire_map)
