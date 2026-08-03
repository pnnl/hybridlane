# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Custom operator attributes"""

from pennylane.ops.qubit import attributes
from pennylane.ops.qubit.attributes import Attribute

diagonal_in_fock_basis = Attribute(
    [
        "FockStateProjector",
        "NumberOperator",
        "TensorN",
        "Rotation",
        "ConditionalRotation",
        "Fourier",
        "SelectiveNumberArbitraryPhase",
        "Kerr",
        "Identity",
    ]
)

diagonal_in_position_basis = Attribute(
    [
        "QuadX",
        "TwoModeSum",
        "ConditionalTwoModeSum",
        "CubicPhase",
        "Identity",
    ]
)

# Gates that can be composed by element-wise adding the parameters of each gate
#   - SNAP gate merging depends on the hyperparameter `n` and therefore can't be included
#   - Many gates can only be fused if their phase parameters are equal (BS, D, S, etc)
attributes.composable_rotations.update(
    [
        "Rotation",
        "Kerr",
        "CubicPhase",
        "TwoModeSum",
        "ConditionalRotation",
        "ConditionalTwoModeSum",
    ]
)
