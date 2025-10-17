from typing import Any

import pennylane as qml
from pennylane.decomposition import CompressedResourceOp
from pennylane.operation import Operator

import hybridlane as hqml


def qubit_conditioned_resource_rep(
    base_class: type[Operator], base_params: dict[str, Any], num_control_wires: int
):
    """Return a resource representation of a qubit-conditioned operator.

    Args:
        base_class: The base operator class.
        base_params: The parameters of the base operator.
        num_control_wires: The number of control wires.

    Returns:
        Operator: The resource representation of the qubit-conditioned operator.
    """

    # Flatten any nested parity-conditioned operators
    if base_class is hqml.ops.QubitConditioned:
        num_control_wires += base_params["num_control_wires"]
        return qubit_conditioned_resource_rep(
            base_params["base_class"], base_params["base_params"], num_control_wires
        )

    # Use any known conditioned gate identities. This handles the case that we know
    # that CustomGate = QubitConditioned(BaseGate)
    known_decomps = hqml.ops.op_math.qubit_conditioned.base_to_custom_conditioned_op()
    if known_decomp := known_decomps.get((base_class, num_control_wires)):
        return qml.resource_rep(known_decomp)

    # Special instance that's not in base_to_custom_conditioned_op
    if base_class is hqml.ops.Rotation:
        return qml.resource_rep(hqml.ops.ConditionalRotation)

    # Decompose instances of QubitConditioned(gate) where gate itself is equivalent to QubitConditioned(othergate)
    # e.g. QubitConditioned(ConditionalDisplacement) = QubitConditioned(Displacement, control_wires=2)
    known_custom_gates = custom_qubit_controlled_op_to_base()
    if known_custom_gate := known_custom_gates.get(base_class):
        num_control_wires = (
            num_control_wires + base_class.num_wires - known_custom_gate.num_wires
        )
        base_class = known_custom_gate

    return CompressedResourceOp(
        hqml.ops.QubitConditioned,
        {
            "base_class": base_class,
            "base_params": base_params,
            "num_control_wires": num_control_wires,
        },
    )


def custom_qubit_controlled_op_to_base():
    return {
        hqml.ConditionalDisplacement: hqml.Displacement,
        hqml.ConditionalSqueezing: hqml.Squeezing,
        hqml.ConditionalParity: hqml.Fourier,
        hqml.ConditionalTwoModeSqueezing: hqml.TwoModeSqueezing,
        hqml.ConditionalTwoModeSum: hqml.TwoModeSum,
        hqml.ConditionalBeamsplitter: hqml.Beamsplitter,
        qml.IsingZZ: qml.RZ,
        qml.MultiRZ: qml.RZ,
    }
