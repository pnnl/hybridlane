from typing import cast

import numpy as np
import pytest
from pennylane.operation import Operation
from pennylane.ops.qubit.attributes import composable_rotations
from pennylane.tape.qscript import QuantumScript
from pennylane.transforms.optimization.merge_rotations import merge_rotations

import hybridlane as hl
from hybridlane.ops.mixins import FockRepresentation
from hybridlane.wires.type_check import infer_wires


def test_composable_rotations():
    fock_dim = 16
    rng = np.random.default_rng()
    for op_name in filter(lambda x: hasattr(hl, x), composable_rotations):
        op_type = cast(type[Operation], getattr(hl, op_name))

        # Determine wire dimensions through type inspection
        def param_fn():
            return rng.standard_normal(op_type.num_params)  # ty:ignore[no-matching-overload]  # noqa: B023

        wires = tuple(range(op_type.num_wires))  # ty:ignore[invalid-argument-type]
        op = op_type(*param_fn(), wires=wires)
        res = infer_wires(op, {})
        wire_dims = {w: 2 if isinstance(t, hl.Qubit) else fock_dim for w, t in res.items()}

        # Make a tape and try merging the rotations
        tape = QuantumScript(
            [
                op_type(*param_fn(), wires=wires),
                op_type(*param_fn(), wires=wires),
            ]
        )

        # For some reason, merge_rotations actually can mutate the input tape
        (new_tape,), _ = merge_rotations(tape.copy(copy_operations=True))
        assert len(new_tape) == 1
        assert isinstance(new_tape[0], op_type)

        # Now we'll compare the matrices before and after to check they're very close
        op1, op2 = (cast(FockRepresentation, x) for x in tape.operations)
        mat1 = op1.fock_matrix(wire_dims)
        mat2 = op2.fock_matrix(wire_dims)
        expected_mat = mat2 @ mat1  # ty:ignore[unsupported-operator]

        actual_mat = hl.fock_matrix(new_tape, wires, wire_dims)
        assert actual_mat == pytest.approx(expected_mat)
