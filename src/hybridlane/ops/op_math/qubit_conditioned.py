# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Qubit-conditioned symbolic operator"""

import itertools
from collections.abc import Callable
from functools import wraps
from typing import ClassVar

import pennylane as qp
from pennylane.operation import Operator
from pennylane.ops.op_math import SymbolicOp
from pennylane.wires import Wires, WiresLike

import hybridlane as hl


def qcond(op: Operator | Callable, control_wires: WiresLike):
    r"""Creates a qubit-conditioned operator

    For a unitary gate, this is the symbolic map :math:`e^{-i\theta G} \mapsto
    e^{-i\theta G \otimes_q Z_q}` where :math:`q` enumerates the qubit control wires. For a
    general callable, this creates a wrapper that applies the function and then applies the
    qubit-conditioned version of all operators in the resulting tape.

    **Example**

    >>> hl.qcond(hl.D(0.123, 0, wires="m"), control_wires="q")
    ConditionalDisplacement(0.123, 0, wires=['q', 'm'])

    This also works with some qubit gates:

    >>> hl.qcond(qp.GlobalPhase(0.123), control_wires=1)
    RZ(0.246, wires=[1])
    >>> hl.qcond(qp.RZ(0.123, wires=0), control_wires=1)
    IsingZZ(0.123, wires=[1, 0])
    """
    return _create_qubit_conditioned_op(op, control_wires)


def _create_qubit_conditioned_op(op: Operator | Callable, control: WiresLike):
    control_wires = Wires(control)

    # Try wrapping in a custom known gate
    key = (type(op), len(control_wires))
    decomps = base_to_custom_conditioned_op()
    if cond_op := decomps.get(key):
        qp.QueuingManager.remove(op)
        return cond_op(*op.data, control_wires + op.wires)  # ty:ignore[unresolved-attribute]

    # Special case because parameter convention change
    if isinstance(op, hl.Rotation) and len(control_wires) == 1:
        qp.QueuingManager.remove(op)
        return hl.ConditionalRotation(2 * op.data[0], control_wires + op.wires)  # ty:ignore[unsupported-operator]

    if isinstance(op, (qp.GlobalPhase, qp.RZ, qp.IsingZZ, qp.MultiRZ)):
        qp.QueuingManager.remove(op)
        return _handle_z_rotations(op, control_wires)

    # Nested qubit condition ops
    if isinstance(op, QubitConditioned):
        control_wires = control_wires + op.control_wires
        qp.QueuingManager.remove(op)
        return qcond(op.base, control_wires)

    if isinstance(op, Operator):
        return QubitConditioned(op, control_wires)

    # Handle qp capture stuff later

    if not callable(op):
        raise ValueError(f"Expected an Operator or Callable, got {type(op)}")

    return _qcond_transform(op, control_wires)


def _qcond_transform(func, control: Wires):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tape = qp.tape.make_qscript(func)(*args, **kwargs)

        leaves, _ = qp.pytrees.flatten((args, kwargs), lambda obj: isinstance(obj, Operator))
        for leaf in leaves:
            if isinstance(leaf, Operator):
                qp.QueuingManager.remove(leaf)

        for op in tape.operations:
            qcond(op, control)

        if qp.QueuingManager.recording():
            for m in tape.measurements:
                qp.apply(m)

        return tape.measurements

    return wrapper


def _handle_z_rotations(op: qp.GlobalPhase | qp.RZ | qp.IsingZZ | qp.MultiRZ, control_wires: Wires):
    param = op.data[0]
    if isinstance(op, qp.GlobalPhase):
        wires = control_wires
        param = 2 * param  # ty:ignore[unsupported-operator]
    else:
        wires = control_wires + op.wires

    new_type = {1: qp.RZ, 2: qp.IsingZZ}
    if new_op_type := new_type.get(len(wires)):
        return new_op_type(param, wires)

    return qp.MultiRZ(param, wires)


class QubitConditioned(SymbolicOp):
    r"""Symbolic operator denoting a qubit-conditioned operator

    For a unitary gate :math:`U = e^{-i\theta G}`, the qubit-conditioned version is

    .. math::

        U = e^{-i\theta G \otimes_q Z_q}

    where :math:`q` enumerates the qubit control wires. This operator is represented symbolically in
    the decomposition system as ``qCond(.)``
    """

    resource_keys: ClassVar = {"base_class", "base_params", "num_control_wires"}

    def _flatten(self):
        return (self.base,), (self.control_wires,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(base=data[0], control_wires=metadata[0])

    @classmethod
    def _primitive_bind_call(
        cls,
        base,
        control_wires,
        id=None,  # noqa: ARG003
    ):
        control_wires = Wires(control_wires)
        return cls._primitive.bind(base, *control_wires)  # ty:ignore[unresolved-attribute]

    def __init__(self, base: Operator, control_wires: WiresLike, id: str | None = None):
        """Construct a qubit-conditioned version of the operator

        Args:
            base: The operator to be conditioned

            control_wires: The qubits to condition the operator on

            id: The id of the operator
        """
        control_wires = Wires(control_wires)

        if base.wires & control_wires:
            raise ValueError("The control wires must be different from the operator wires")

        self.hyperparameters["control_wires"] = control_wires
        self.name: str = f"QubitConditioned({base.name})"

        super().__init__(base, id)

    @property
    def control_wires(self) -> Wires:
        r"""The qubit wires that the operator is conditioned on"""
        return self.hyperparameters["control_wires"]

    @property
    def wires(self) -> Wires:
        r"""The wires that the operator acts on, including the control wires"""
        return self.control_wires + self.base.wires

    @property
    def resource_params(self):  # noqa: D102
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params,
            "num_control_wires": len(self.control_wires),
        }

    def __repr__(self):
        params = [f"control_wires={self.control_wires.tolist()}"]
        return f"QubitConditioned({self.base}, {', '.join(params)})"

    def label(self, decimals: int | None = None, base_label: str | None = None, cache=None):  # noqa: D102
        return self.base.label(decimals=decimals, base_label=base_label, cache=cache)

    @property
    def has_diagonalizing_gates(self):  # noqa: D102
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self) -> list[Operator]:  # noqa: D102
        return super().diagonalizing_gates()

    @property
    def has_decomposition(self):  # noqa: D102
        if self.compute_decomposition is not Operator.compute_decomposition:
            return True

        # We can use cnots to eliminate all but one control wire
        if len(self.control_wires) > 1:
            return True

        known_decomps = base_to_custom_conditioned_op()
        if (type(self.base), len(self.control_wires)) in known_decomps:
            return True

        if type(self.base) in (qp.GlobalPhase, qp.Identity, hl.Rotation):
            return True

        return bool(
            len(self.control_wires) == 1
            and hasattr(self.base, "_qubit_conditioned")
            and type(self) is QubitConditioned
        )

    def decomposition(self):  # noqa: D102
        if self.compute_decomposition is not Operator.compute_decomposition:
            return self.compute_decomposition(*self.data, self.wires)

        if (decomp := _decompose_custom_op(self)) is None:
            raise qp.decomposition.DecompositionUndefinedError(  # ty:ignore[unresolved-attribute]
                f"Decomposition not defined for {self}"
            )

        return decomp

    @property
    def has_generator(self):  # noqa: D102
        return self.base.has_generator

    def generator(self):  # noqa: D102
        z_factors = [qp.Z(w) for w in self.control_wires]
        return qp.prod(*z_factors, self.base.generator())

    @property
    def has_adjoint(self):  # noqa: D102
        return self.base.has_adjoint

    def adjoint(self):  # noqa: D102
        return QubitConditioned(self.base.adjoint(), self.control_wires)

    def pow(self, z):  # noqa: D102
        return QubitConditioned(qp.pow(self.base, z), self.control_wires)

    def __eq__(self, other):
        if not isinstance(other, QubitConditioned):
            return False
        return self.base == other.base and self.control_wires == other.control_wires


def _decompose_custom_op(op: QubitConditioned) -> list[Operator] | None:
    custom_decomps = base_to_custom_conditioned_op()
    custom_key = (type(op.base), len(op.control_wires))

    if custom_decomp := custom_decomps.get(custom_key):
        return [custom_decomp(*op.data, wires=op.wires)]

    # We just add more Zs
    if isinstance(op.base, qp.MultiRZ):
        return [qp.MultiRZ(*op.base.data, wires=op.control_wires + op.base.wires)]

    # Conditioned version of identity is identity as I = exp(-i 0 I), so exp(-i 0 ZI) = I
    if isinstance(op.base, qp.Identity):
        return [qp.Identity(op.control_wires + op.base.wires)]

    if isinstance(op.base, qp.GlobalPhase):
        return [qp.MultiRZ(2 * op.base.data[0], wires=op.control_wires)]  # ty:ignore[unsupported-operator]

    # We can always use CNOTs to take a single Z in the generator and extend it to arbitrary qubits
    if len(op.control_wires) >= 2:
        cnots = [qp.CNOT(wires=(c, t)) for c, t in itertools.pairwise(op.control_wires)]
        return [*cnots, qcond(op.base, [op.control_wires[-1]]), *cnots[::-1]]

    # Handle the differing factor of 2 in the definitions
    if isinstance(op.base, hl.Rotation):
        return [
            hl.ConditionalRotation(
                2 * op.base.data[0],  # ty:ignore[unsupported-operator]
                op.control_wires + op.base.wires,
            )
        ]

    return None


# Dictionary mapping operators to their conditional versions, if the parameters are the same
def base_to_custom_conditioned_op() -> dict[tuple[type[Operator], int], type[Operator]]:
    r"""Returns a dictionary mapping base operators to their conditional versions

    The keys are tuples of the form ``(base_class, num_control_wires)`` and the values are the
    corresponding conditional operator classes. This is used to determine if a custom conditional
    operator exists for a given base operator and number of control wires.
    """
    return {
        (hl.Displacement, 1): hl.ConditionalDisplacement,
        (hl.Fourier, 1): hl.ConditionalParity,
        (hl.Squeezing, 1): hl.ConditionalSqueezing,
        (hl.Beamsplitter, 1): hl.ConditionalBeamsplitter,
        (hl.TwoModeSqueezing, 1): hl.ConditionalTwoModeSqueezing,
        (hl.TwoModeSum, 1): hl.ConditionalTwoModeSum,
        (qp.RZ, 1): qp.IsingZZ,
        (qp.IsingZZ, 1): qp.MultiRZ,
    }


if QubitConditioned._primitive is not None:

    @QubitConditioned._primitive.def_impl
    def _(base, *control_wires, id=None):
        return type.__call__(QubitConditioned, base, control_wires, id=id)
