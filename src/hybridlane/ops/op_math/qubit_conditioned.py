import pennylane as qml
from pennylane.ops.op_math import SymbolicOp
from pennylane.operation import Operator
from pennylane.wires import Wires, WiresLike

import hybridlane as hqml
from hybridlane.decomposition.resources import qubit_conditioned_resource_rep


def qcond(op: Operator, control_wires: WiresLike):
    return QubitConditioned(op, control_wires)


class QubitConditioned(SymbolicOp):
    r"""Symbolic operator denoting a qubit-conditioned operator

    For a unitary gate :math:`U = e^{-i\theta G}`, the qubit-conditioned version is

    .. math::

        U = e^{-i\theta G \otimes_q Z_q}

    where :math:`q` enumerates the qubit control wires. This operator is represented symbolically in
    the decomposition system as ``qCond(.)``
    """

    resource_keys = {"base_class", "base_params", "num_control_wires"}

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
        id=None,
    ):
        control_wires = Wires(control_wires)
        return cls._primitive.bind(base, *control_wires)

    def __init__(self, base: Operator, control_wires: WiresLike, id: str | None = None):
        """
        Args:
            base: The operator to be conditioned
            control_wires: The qubits to condition the operator on
            id: The id of the operator
        """
        control_wires = Wires(control_wires)

        if base.wires & control_wires:
            raise ValueError(
                "The control wires must be different from the operator wires"
            )

        self.hyperparameters["control_wires"] = control_wires
        self.name: str = f"qCond({base.name})"

        super().__init__(base, id)

    @property
    def control_wires(self) -> Wires:
        return self.hyperparameters["control_wires"]

    @property
    def wires(self) -> Wires:
        return self.control_wires + self.base.wires

    def resource_params(self):
        return {
            "base_class": type(self.base),
            "base_params": self.base.resource_params(),
            "num_control_wires": len(self.control_wires),
        }

    def __repr__(self):
        params = [f"control_wires={self.control_wires.tolist()}"]
        return f"QubitConditioned({self.base}, {', '.join(params)})"

    def label(
        self, decimals: int | None = None, base_label: str | None = None, cache=None
    ):
        return self.base.label(decimals=decimals, base_label=base_label, cache=cache)

    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self) -> list[Operator]:
        return super().diagonalizing_gates()

    @property
    def has_decomposition(self):
        if self.compute_decomposition is not Operator.compute_decomposition:
            return True

        # We can use cnots to eliminate all but one control wire
        if len(self.control_wires) > 1:
            return True

        known_decomps = base_to_custom_conditioned_op()
        if (type(self.base), len(self.control_wires)) in known_decomps:
            return True

        if type(self.base) in (qml.GlobalPhase, qml.Identity, hqml.Rotation):
            return True

        if (
            len(self.control_wires) == 1
            and hasattr(self.base, "_qubit_conditioned")
            and type(self) is QubitConditioned
        ):
            return True

        return False

    def decomposition(self):
        if self.compute_decomposition is not Operator.compute_decomposition:
            return self.compute_decomposition(*self.data, self.wires)

        if (decomp := _decompose_custom_op(self)) is None:
            raise qml.decomposition.DecompositionUndefinedError(
                f"Decomposition not defined for {self}"
            )

        return decomp

    @property
    def has_generator(self):
        return self.base.has_generator

    def generator(self):
        z_factors = [qml.Z(w) for w in self.control_wires]
        return qml.prod(*z_factors, self.base.generator())

    @property
    def has_adjoint(self):
        return self.base.has_adjoint

    def adjoint(self):
        return QubitConditioned(self.base.adjoint(), self.control_wires)

    def pow(self, z):
        return QubitConditioned(qml.pow(self.base, z), self.control_wires)

    def __eq__(self, other):
        if not isinstance(other, QubitConditioned):
            return False
        return self.base == other.base and self.control_wires == other.control_wires


def _qcond_cnot_decomposition_resources(base_class, base_params, num_control_wires):
    return {
        qubit_conditioned_resource_rep(base_class, base_params, 1): 1,
        qml.CNOT: 2 * (num_control_wires - 1),
    }


@qml.register_condition(lambda num_control_wires: num_control_wires > 1)
@qml.register_resources(_qcond_cnot_decomposition_resources)
def _qcond_cnot_decomposition(wires, base, control_wires, **_):
    ct = list(zip(control_wires[:-1], control_wires[1:]))
    for c, t in ct:
        qml.CNOT(wires=[c, t])

    hqml.qcond(base, control_wires[-1])

    for c, t in reversed(ct):
        qml.CNOT(wires=[c, t])


qml.add_decomps(QubitConditioned, _qcond_cnot_decomposition)
# Todo: What about pow/adjoint(qCond)?


def _decompose_custom_op(op: QubitConditioned) -> list[Operator] | None:
    custom_decomps = base_to_custom_conditioned_op()
    custom_key = (type(op.base), len(op.control_wires))

    if custom_decomp := custom_decomps.get(custom_key):
        return [custom_decomp(*op.data, wires=op.wires)]

    # We just add more Zs
    if isinstance(op.base, qml.MultiRZ):
        return [qml.MultiRZ(*op.base.data, wires=op.control_wires + op.base.wires)]

    # Conditioned version of identity is identity as I = exp(-i 0 I), so exp(-i 0 ZI) = I
    if isinstance(op.base, qml.Identity):
        return [qml.Identity(op.control_wires + op.base.wires)]

    if isinstance(op.base, qml.GlobalPhase):
        return [qml.MultiRZ(2 * op.base.data[0], wires=op.control_wires)]

    # We can always use CNOTs to take a single Z in the generator and extend it to arbitrary qubits
    if len(op.control_wires) >= 2:
        cnots = [
            qml.CNOT(wires=(c, t))
            for c, t in zip(op.control_wires[:-1], op.control_wires[1:])
        ]
        return cnots + [QubitConditioned(op.base, [op.control_wires[-1]])] + cnots[::-1]

    # Handle the differing factor of 2 in the definitions
    if isinstance(op.base, hqml.Rotation):
        return [
            hqml.ConditionalRotation(
                2 * op.base.data[0], op.control_wires + op.base.wires
            )
        ]

    # Here pennylane has a _controlled attribute defined, but then they marked it as to be removed. I can't
    # find where they define [sc-37951] to talk about their broader refactoring plans

    return None


# Dictionary mapping operators to their conditional versions, if the parameters are the same
def base_to_custom_conditioned_op() -> dict[tuple[type[Operator], int], type[Operator]]:
    return {
        (hqml.Displacement, 1): hqml.ConditionalDisplacement,
        (hqml.Fourier, 1): hqml.ConditionalParity,
        (hqml.Squeezing, 1): hqml.ConditionalSqueezing,
        (hqml.Beamsplitter, 1): hqml.ConditionalBeamsplitter,
        (hqml.TwoModeSqueezing, 1): hqml.ConditionalTwoModeSqueezing,
        (hqml.TwoModeSum, 1): hqml.ConditionalTwoModeSum,
        (qml.RZ, 1): qml.IsingZZ,
        (qml.IsingZZ, 1): qml.MultiRZ,
    }


if QubitConditioned._primitive is not None:

    @QubitConditioned._primitive.def_impl
    def _(base, *control_wires, id=None):
        return type.__call__(QubitConditioned, base, control_wires, id=id)
