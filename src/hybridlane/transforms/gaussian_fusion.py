# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from functools import reduce

import pennylane as qml
from pennylane.operation import CVOperation, Operator
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.typing import PostprocessingFn
from pennylane.wires import Wires
from typing_extensions import TypeIs

import hybridlane as hqml


@qml.transform
def gaussian_fusion(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Fuses Gaussian CV gates across qumodes into a single multi-qumode Gaussian gate"""
    new_ops: list[Operator] = []

    to_fuse: list[CVOperation] = []
    curr_wires = Wires([])
    for op in tape.operations:
        if _is_gaussian(op):
            to_fuse.append(op)
            curr_wires = curr_wires | op.wires
            continue

        # Non-gaussian operation. If it commutes with all the current gaussian
        # gates in this block, then we can just queue it and keep fusing. Otherwise, this marks the
        # end of the block and we fuse everything that we've accumulated.
        if (op.wires & curr_wires) and to_fuse:
            new_ops.append(_fuse_gaussian(to_fuse))
            to_fuse.clear()
            curr_wires = Wires([])

        new_ops.append(op)

    if to_fuse:
        new_ops.append(_fuse_gaussian(to_fuse))

    new_tape = QuantumScript(new_ops, tape.measurements)

    def null_postprocessing(results):
        return results[0]

    return (new_tape,), null_postprocessing


def _is_gaussian(op: Operator) -> TypeIs[CVOperation]:
    return isinstance(op, CVOperation) and op.supports_heisenberg


def _fuse_gaussian(ops: list[CVOperation]) -> CVOperation:
    if len(ops) == 0:
        raise RuntimeError("Cannot fuse an empty list of gates")

    wire_order = Wires.all_wires([op.wires for op in ops])
    mats = []

    for op in ops:
        # This returns the wires of `op`, but in the order of `wire_order`
        local_wire_order = Wires.shared_wires([wire_order, op.wires])
        S = op.heisenberg_tr(local_wire_order)

        # Expand to match the full system dimension
        S = op.heisenberg_expand(S, wire_order)

        # Put in the proper order for matmul
        mats.insert(0, S)

    S = reduce(lambda x, y: x @ y, mats)
    return hqml.Gaussian(S, wire_order)
