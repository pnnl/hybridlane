# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Construct fock matrix representations of operators and quantum functions"""

import functools
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, cast, overload

import pennylane as qp
from pennylane.exceptions import TransformError
from pennylane.operation import Operator
from pennylane.pauli.pauli_arithmetic import PauliSentence, PauliWord
from pennylane.tape.qscript import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn, TensorLike
from pennylane.wires import Wires, WiresLike
from pennylane.workflow.qnode import QNode

import hybridlane as hl

from ..mixins import FockRepresentation

TransformOutput = tuple[QuantumScriptBatch, PostprocessingFn]


@overload
def fock_matrix(
    op: Callable | type[Operator] | Sequence[Operator] | QuantumScript,
    wire_order: WiresLike | None = None,
    wire_dims: Mapping[Any, int] | None = None,
) -> Callable[..., TransformOutput]: ...


@overload
def fock_matrix(
    op: Operator | PauliWord | PauliSentence,
    wire_order: WiresLike | None = None,
    wire_dims: Mapping[Any, int] | None = None,
) -> TensorLike: ...


def fock_matrix(
    op: QNode
    | Callable
    | PauliWord
    | PauliSentence
    | Operator
    | Sequence[Operator]
    | QuantumScript
    | type[Operator],
    wire_order: WiresLike | None = None,
    wire_dims: Mapping[Any, int] | None = None,
) -> TensorLike | Callable[..., TransformOutput]:
    r"""Compute the matrix representation in the Fock basis.

    Like :func:`~pennylane.matrix`, this transform can be applied to many types including
    operators, tapes, and quantum functions. It differs from the original by also requiring a
    ``wire_dims`` argument, which is a mapping of wire labels to their corresponding Hilbert space
    dimensions.

    Args:
        op: The operator, tape, or quantum function to compute the matrix representation of.

        wire_order: The order of the wires in the resulting matrix

        wire_dims: A mapping of wire labels to their corresponding Hilbert space dimensions.

    Returns:
        If ``op`` is an operator, it returns the matrix. Otherwise, it acts like a transform

    .. seealso:: :func:`~pennylane.matrix`

    **Example**

    It can be used to create a new function that returns the matrix of a quantum function.

    .. code-block:: python

        def test_fn(theta):
            qp.X(0)
            hl.CR(theta, wires=(0, 1))
            qp.X(0)

            return hl.expval(hl.N(1))

    >>> matrix_fn = hl.fock_matrix(test_fn, wire_order=(0, 1), wire_dims={0: 2, 1: 3})
    >>> matrix_fn(0.123)
    array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
            0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.9981+0.0615j, 0.    +0.j    , 0.    +0.j    ,
            0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.9924+0.1227j, 0.    +0.j    ,
            0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 1.    +0.j    ,
            0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
            0.9981-0.0615j, 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ,
            0.    +0.j    , 0.9924-0.1227j]])

    It can obtain the matrix for a single operator:

    >>> hl.fock_matrix(hl.K(0.123, wires=0), wire_dims={0: 4})
    array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.9924-0.1227j, 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.8814-0.4724j, 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.4473-0.8944j]])

    Like ``qp.matrix``, this can also be done in a functional form:

    >>> hl.fock_matrix(hl.K, wire_dims={0: 4}, wire_order=[0])(0.123, wires=0)
    array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.9924-0.1227j, 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.8814-0.4724j, 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.4473-0.8944j]])
    """
    # Everything that can be coerced into a program
    if not isinstance(op, (Operator, PauliWord, PauliSentence)):
        if wire_dims is None:
            raise ValueError(
                "`wire_dims` must be specified for the fock_matrix transform "
                "when applied to quantum functions"
            )

        input = op
        match op:
            case Sequence():
                input = QuantumScript(op)  # ty:ignore[invalid-argument-type]

            case QuantumScript():
                if wire_order is None and len(op.wires) != 1:
                    raise ValueError(
                        "`wire_order` must be specified for tapes with zero or more than one wire."
                    )

            case QNode(device=dev):
                if wire_order is None and dev.wires is None:
                    raise ValueError(
                        "`wire_order` is required when the device does not have a wire order"
                    )

            case op if callable(op):
                # the getattr part handles type[operator]
                if getattr(op, "num_wires", 0) != 1 and wire_order is None:
                    raise ValueError("`wire_order` must be specified for callables")

            case _:  # pragma: no cover
                raise TransformError(
                    f"The provided input type {type(op)} is not compatible with "
                    "the fock_matrix transform."
                )

        return _fock_matrix_transform(input, wire_dims, wire_order)

    # At this point, it's an operator or pauli word or sentence that we can construct the matrix
    # for
    if wire_order is not None and not set(op.wires).issubset(wire_order):  # ty:ignore[invalid-argument-type]
        raise TransformError(
            "The provided `wire_order` does not contain all the wires of the operator."
        )

    match op:
        case FockRepresentation():
            if wire_dims is None:
                raise ValueError(
                    "`wire_dims` must be specified for the fock_matrix "
                    "when applied to operators with a CV component"
                )

            return op.fock_matrix(wire_order=wire_order, wire_dims=wire_dims)

        case _:
            # We need to subtract all the non-qubit wires from this operator then manually
            # expand the matrix back out, otherwise PennyLane will default to dimension 2
            # for all the wires and that might be incorrect
            mat = qp.matrix(op, wire_order=op.wires)

            if wire_order is None:
                return mat

            if wire_dims is None and (missing_wires := Wires(wire_order) - op.wires):
                raise ValueError(
                    "`wire_dims` must be specified with wire_order when applied to operators. "
                    f"we have no way of knowing what dimension wires {missing_wires} should be"
                )

            return hl.math.expand_matrix(mat, op.wires, wire_order=wire_order, wire_dims=wire_dims)


@partial(qp.transform, is_informative=True)
def _fock_matrix_transform(
    tape: QuantumScript,
    wire_dims: Mapping[Any, int] | None = None,
    wire_order: WiresLike | None = None,
    **kwargs,
) -> TransformOutput:
    # First type check the tape to infer dimensions for qubits and qumodes if necessary
    wire_dims = _validate_wire_dims(tape, wire_dims or {})

    # Now we use similar logic to qp.matrix, performing matmul reduction over the tape operations
    wires = kwargs.get("device_wires") or tape.wires
    wire_order = Wires(wire_order or wires)

    def matmul(u: TensorLike, op: Operator, like=None):
        v = fock_matrix(op, wire_order=wire_order, wire_dims=wire_dims)
        u, v = qp.math.coerce([u, v], like=like)
        return qp.math.matmul(v, u, like=like)  # V @ U

    def postprocessing_fn(res: QuantumScriptBatch) -> TensorLike:
        tape, *_ = res
        like = qp.math.get_deep_interface(tape.get_parameters(trainable_only=False))
        hilbert_dim = int(qp.math.prod([wire_dims[w] for w in wire_order]))  # ty:ignore[not-subscriptable]
        id = qp.math.eye(hilbert_dim, like=like)
        return functools.reduce(partial(matmul, like=like), tape.operations, id)

    return (tape,), postprocessing_fn


def _validate_wire_dims(tape: QuantumScript, wire_dims: Mapping[Any, int]) -> dict[Any, int]:
    wire_dims = dict(wire_dims)

    res = hl.type_check(tape)
    for wire, wire_type in res.wire_types.items():
        # Validate every qumode has a dimension provided
        if wire_type == hl.Qumode():
            if wire not in wire_dims:
                raise TransformError(
                    "Quantum function has a qumode but no dimension was provided through wire_dims"
                )

        # Validate that the provided wire_dims are consistent with the type check results
        else:
            expected_dim = 2 if wire_type == hl.Qubit() else cast(hl.Qudit, wire_type).dim
            if (found_dim := wire_dims.setdefault(wire, expected_dim)) != expected_dim:
                raise TransformError(
                    f"Wire {wire} is of type {wire_type} but has dimension "
                    f"{found_dim} in wire_dims. Expected dimension: {expected_dim}"
                )

    return wire_dims
