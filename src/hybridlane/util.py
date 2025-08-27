from pennylane.operation import Operator
from pennylane.ops import Prod, Sum, SymbolicOp


def is_tensor_product(obs: Operator) -> bool:
    if isinstance(obs, SymbolicOp):
        return is_tensor_product(obs.base)

    elif isinstance(obs, Sum):
        coeffs, ops = obs.terms()
        if len(coeffs) == 1:
            return is_tensor_product(ops[0])

        return False

    elif isinstance(obs, Prod):
        return not obs.has_overlapping_wires and all(
            is_tensor_product(op) for op in obs.operands
        )

    else:
        return True
