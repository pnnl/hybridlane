# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

__all__ = [
    "merge_qubit_conditioned",
    "flip_pow_qcond",
    "ctrl_from_qcond",
    "make_qcond_decomp",
]

import pennylane as qml
from pennylane.decomposition import DecompositionRule
from pennylane.decomposition.resources import adjoint_resource_rep, pow_resource_rep

import hybridlane as hqml
from hybridlane.decomposition.resources import qubit_conditioned_resource_rep

# Qubit-conditioned operator decompositions


def _merge_qcond_resources(base_class, base_params, num_control_wires):
    return qubit_conditioned_resource_rep(
        base_params["base_class"],
        base_params["base_params"],
        num_control_wires + base_params["num_control_wires"],
    )


@qml.register_resources(_merge_qcond_resources)
def merge_qubit_conditioned(*params, wires, base, control_wires, **_):
    """Flattens a nested Qubit-conditioned operator"""
    base_op = base.base._unflatten(*base.base._flatten())
    hqml.qcond(base_op, control_wires + base.control_wires)


def _flip_pow_qcond_resources(base_class, base_params, z):
    target_class, target_params = base_params["base_class"], base_params["base_params"]
    return {
        qubit_conditioned_resource_rep(
            qml.ops.Pow,
            {"base_class": target_class, "base_params": target_params, "z": z},
            base_params["num_control_wires"],
        ): 1
    }


@qml.register_resources(_flip_pow_qcond_resources)
def flip_pow_qcond(*params, wires, z, base, **_):
    # Base is QubitConditioned
    base_op = base.base._unflatten(*base.base._flatten())
    control_wires = base.control_wires
    hqml.qcond(qml.pow(base_op, z), control_wires)


def _ctrl_from_qcond_resources(base_class, base_params, num_control_wires, **_):
    qcond_rep = qubit_conditioned_resource_rep(
        base_class, base_params, num_control_wires
    )
    pow_qcond_rep = pow_resource_rep(qcond_rep.op_type, qcond_rep.params, 0.5)
    adj_pow_qcond_rep = adjoint_resource_rep(
        pow_qcond_rep.op_type, pow_qcond_rep.params
    )

    return {pow_resource_rep(base_class, base_params, 0.5): 1, adj_pow_qcond_rep: 1}


@qml.register_condition(
    lambda num_control_wires, num_zero_control_values, **_: num_control_wires == 1
    and num_zero_control_values == 0
)
@qml.register_resources(_ctrl_from_qcond_resources)
def ctrl_from_qcond(*params, wires, base, control_wires, **_):
    r"""Synthesizes a qubit-controlled gate using the qubit-conditioned gate

    This implements the identity

    .. math::

        cU = \sqrt{U} \sqrt{CU}^\dagger
    """
    base_op = base._unflatten(*base._flatten())
    qml.pow(base_op, 0.5)
    qml.adjoint(qml.pow(hqml.qcond(base_op, control_wires), 0.5))


def make_qcond_decomp(base_decomposition: DecompositionRule) -> DecompositionRule:
    def _condition_fn(base_params, **_):
        return base_decomposition.is_applicable(**base_params)

    def _resource_fn(base_params, num_control_wires, **_):
        base_resources = base_decomposition.compute_resources(**base_params)
        gate_counts = {
            qubit_conditioned_resource_rep(
                base_class=base_op_rep.op_type,
                base_params=base_op_rep.params,
                num_control_wires=num_control_wires,
            ): count
            for base_op_rep, count in base_resources.gate_counts.items()
        }
        return gate_counts

    @qml.register_condition(_condition_fn)
    @qml.register_resources(_resource_fn, exact=base_decomposition.exact_resources)
    def _impl(*params, wires, control_wires, base, **_):
        hqml.qcond(base_decomposition._impl, control_wires=wires[: len(control_wires)])(
            *params, wires=wires[-len(base.wires) :], **base.hyperparameters
        )

    return _impl
