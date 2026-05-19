# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pennylane as qp
from pennylane.operation import Operation

import hybridlane as hl

from ....decomposition.resources import qubit_conditioned_resource_rep


def to_native_qcond(control_wires: int):
    """Decomposition rule for qcond of gates that have a native equivalent defined in hl.qcond"""

    def _condition_fn(num_control_wires, **_):
        return num_control_wires == control_wires

    def _resource_fn(base_class, base_params, num_control_wires):
        return {
            qubit_conditioned_resource_rep(
                base_class, base_params, num_control_wires
            ): 1
        }

    @qp.register_condition(_condition_fn)
    @qp.register_resources(_resource_fn)
    def _impl(*params, wires, base, control_wires, **_):
        base_op = base._unflatten(*base._flatten())
        hl.qcond(base_op, control_wires)

    return _impl


def _decompose_multiqcond_native_resources(base_class, base_params, num_control_wires):
    return {
        qp.resource_rep(base_class, **base_params): 1,
        qp.CNOT: 2 * num_control_wires,
    }


@qp.register_resources(_decompose_multiqcond_native_resources)
def decompose_multiqcond_native(*params, wires, base, control_wires, **_):
    control_wires = control_wires + base.wires[0]

    ct = list(zip(control_wires[:-1], control_wires[1:]))
    for c, t in ct:
        qp.CNOT(wires=[c, t])

    qp.apply(base)

    for c, t in reversed(ct):
        qp.CNOT(wires=[c, t])


def _decompose_multi_qcond_resources(base_class, base_params, num_control_wires):
    return {
        qubit_conditioned_resource_rep(base_class, base_params, 1): 1,
        qp.CNOT: 2 * (num_control_wires - 1),
    }


@qp.register_condition(lambda num_control_wires, **_: num_control_wires > 1)
@qp.register_resources(_decompose_multi_qcond_resources)
def decompose_multi_qcond(*params, wires, base, control_wires, **_):
    base_op = base._unflatten(*base._flatten())

    ct = list(zip(control_wires[:-1], control_wires[1:]))
    for c, t in ct:
        qp.CNOT(wires=[c, t])

    hl.qcond(base_op, control_wires[-1])

    for c, t in reversed(ct):
        qp.CNOT(wires=[c, t])


def make_gate_with_ancilla_qubit(base_class: type[Operation]):
    r"""Synthesizes a gate using an ancilla qubit and qubit conditioned operations

    For example, this decomposition rule allows for creating a displacement gate
    :math:`D_m(\alpha)` by allocating an ancilla qubit :math:`a` and applying
    :math:`CD_{a,m}(\alpha)` instead. This is a particularly useful rule on platforms
    that don't support native bosonic gates (e.g. ion traps).
    """

    def _resource_fn(**base_params):
        return {
            qubit_conditioned_resource_rep(
                base_class, base_params, num_control_wires=1
            ): 1
        }

    @qp.register_resources(_resource_fn, work_wires={"zeroed": 1})
    def _impl(*params, wires, **hparams):
        with qp.allocate(1, "zero", restored=True) as ancilla:
            hl.qcond(base_class, control_wires=ancilla)(*params, wires=wires, **hparams)

    return _impl
