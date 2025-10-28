# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import pennylane as qml

import hybridlane as hqml

from ....decomposition.resources import qubit_conditioned_resource_rep


def to_native_qcond(control_wires: int):
    """Decomposition rule for qcond of gates that have a native equivalent defined in hqml.qcond"""

    def _condition_fn(num_control_wires, **_):
        return num_control_wires == control_wires

    def _resource_fn(base_class, base_params, num_control_wires):
        return {
            qubit_conditioned_resource_rep(
                base_class, base_params, num_control_wires
            ): 1
        }

    @qml.register_condition(_condition_fn)
    @qml.register_resources(_resource_fn)
    def _impl(*params, wires, base, control_wires, **_):
        base_op = base._unflatten(*base._flatten())
        hqml.qcond(base_op, control_wires)

    return _impl


def _decompose_multiqcond_native_resources(base_class, base_params, num_control_wires):
    return {
        qml.resource_rep(base_class, **base_params): 1,
        qml.CNOT: 2 * num_control_wires,
    }


@qml.register_resources(_decompose_multiqcond_native_resources)
def decompose_multiqcond_native(*params, wires, base, control_wires, **_):
    control_wires = control_wires + base.wires[0]

    ct = list(zip(control_wires[:-1], control_wires[1:]))
    for c, t in ct:
        qml.CNOT(wires=[c, t])

    qml.apply(base)

    for c, t in reversed(ct):
        qml.CNOT(wires=[c, t])


def _decompose_multi_qcond_resources(base_class, base_params, num_control_wires):
    return {
        qubit_conditioned_resource_rep(base_class, base_params, 1): 1,
        qml.CNOT: 2 * (num_control_wires - 1),
    }


@qml.register_condition(lambda num_control_wires, **_: num_control_wires > 1)
@qml.register_resources(_decompose_multi_qcond_resources)
def decompose_multi_qcond(*params, wires, base, control_wires, **_):
    base_op = base._unflatten(*base._flatten())

    ct = list(zip(control_wires[:-1], control_wires[1:]))
    for c, t in ct:
        qml.CNOT(wires=[c, t])

    hqml.qcond(base_op, control_wires[-1])

    for c, t in reversed(ct):
        qml.CNOT(wires=[c, t])
