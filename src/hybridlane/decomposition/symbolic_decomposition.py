import pennylane as qml
from pennylane.decomposition.resources import adjoint_resource_rep

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


def _adjoint_qcond_resources(base_class, base_params, num_control_wires):
    adjoint_resource_rep
