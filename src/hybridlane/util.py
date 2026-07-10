# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Module containing various utility functions."""

from pennylane.operation import Operator
from pennylane.ops import Prod, Sum, SymbolicOp


def is_tensor_product(obs: Operator) -> bool:
    r"""Checks if an operator is a tensor product of other operators.

    Args:
        obs: The operator to check.

    Returns:
        True if the operator is a tensor product, False otherwise.
    """
    if isinstance(obs, SymbolicOp):
        return is_tensor_product(obs.base)

    elif isinstance(obs, Sum):
        coeffs, ops = obs.terms()
        if len(coeffs) == 1:
            return is_tensor_product(ops[0])

        return False

    elif isinstance(obs, Prod):
        return not obs.has_overlapping_wires and all(is_tensor_product(op) for op in obs.operands)

    else:
        return True
