# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Patches and related functionality to support resource reps of custom symbolic operators."""

from typing import Any
from unittest.mock import patch

import pennylane as qp
from pennylane.decomposition import CompressedResourceOp
from pennylane.operation import Operator

import hybridlane as hl

# Patch to handle things like qCond(Pow(...))
_qp_resource_rep = qp.resource_rep


def resource_rep(op_type: type[Operator], **params) -> CompressedResourceOp:
    r"""Replacement of ``pennylane.resource_rep`` that handles qubit-conditioned operators.

    Args:
        op_type: The operator class to construct a resource representation for

        params: The resource parameters of the operator (``op.resource_params``)

    Returns:
        The resource representation of the operator
    """
    if issubclass(op_type, hl.ops.QubitConditioned):
        base_rep = resource_rep(params["base_class"], **params["base_params"])
        params["base_class"] = base_rep.op_type
        params["base_params"] = base_rep.params

    return _qp_resource_rep(op_type, **params)


_ = patch("pennylane.decomposition.resources.resource_rep", resource_rep).start()
_ = patch("pennylane.decomposition.decomposition_graph.resource_rep", resource_rep).start()
_ = patch("pennylane.decomposition.symbolic_decomposition.resource_rep", resource_rep).start()


def qubit_conditioned_resource_rep(
    base_class: type[Operator], base_params: dict[str, Any], num_control_wires: int
) -> CompressedResourceOp:
    """Returns a resource representation of a qubit-conditioned operator.

    Args:
        base_class: The base operator class.

        base_params: The parameters of the base operator.

        num_control_wires: The number of control wires.

    Returns:
        Operator: The resource representation of the qubit-conditioned operator.
    """
    # Flatten any nested parity-conditioned operators
    if base_class is hl.ops.QubitConditioned:
        num_control_wires += base_params["num_control_wires"]
        return qubit_conditioned_resource_rep(
            base_params["base_class"], base_params["base_params"], num_control_wires
        )

    # Use any known conditioned gate identities. This handles the case that we know
    # that CustomGate = QubitConditioned(BaseGate)
    known_decomps = hl.ops.op_math.qubit_conditioned.base_to_custom_conditioned_op()
    if known_decomp := known_decomps.get((base_class, num_control_wires)):
        return qp.resource_rep(known_decomp)

    # Special instance that's not in base_to_custom_conditioned_op
    if base_class is hl.ops.Rotation:
        return qp.resource_rep(hl.ops.ConditionalRotation)

    # Decompose instances of QubitConditioned(gate) where gate itself is equivalent to
    # QubitConditioned(othergate) e.g.
    # QubitConditioned(CD) = QubitConditioned(D, control_wires=2)
    known_custom_gates = custom_qubit_controlled_op_to_base()
    if known_custom_gate := known_custom_gates.get(base_class):
        num_control_wires = (
            num_control_wires + base_class.num_wires - known_custom_gate.num_wires  # ty:ignore[unsupported-operator]
        )
        base_class = known_custom_gate
    elif base_class is qp.MultiRZ:  # special exception
        num_control_wires = num_control_wires + base_params["num_wires"] - 1
        base_class = qp.RZ
        base_params = {}

    return CompressedResourceOp(
        hl.ops.QubitConditioned,
        {
            "base_class": base_class,
            "base_params": base_params,
            "num_control_wires": num_control_wires,
        },
    )


def custom_qubit_controlled_op_to_base():
    r"""Returns a mapping of custom qubit-conditioned operations to their unconditioned versions."""
    return {
        hl.ConditionalDisplacement: hl.Displacement,
        hl.ConditionalSqueezing: hl.Squeezing,
        hl.ConditionalParity: hl.Fourier,
        hl.ConditionalTwoModeSqueezing: hl.TwoModeSqueezing,
        hl.ConditionalTwoModeSum: hl.TwoModeSum,
        hl.ConditionalBeamsplitter: hl.Beamsplitter,
        qp.IsingZZ: qp.RZ,
    }


# Patch to support qCond(.) notation
_old_prop = CompressedResourceOp.name
_old_fn = CompressedResourceOp.name.fget


def _cro_name_fget(self: CompressedResourceOp):
    if self.op_type in (hl.ops.QubitConditioned,):
        base_rep = qp.resource_rep(self.params["base_class"], **self.params["base_params"])
        return f"qCond({base_rep.name})"

    return _old_fn(self)


new_name_prop = property(
    fget=_cro_name_fget,
    fset=_old_prop.fset,
    fdel=_old_prop.fdel,
    doc=_old_prop.__doc__,
)

CompressedResourceOp.name = new_name_prop  # ty:ignore[invalid-assignment]
