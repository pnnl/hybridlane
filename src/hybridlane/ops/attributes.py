# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from pennylane.ops.qubit import attributes
from pennylane.ops.qubit.attributes import Attribute

diagonal_in_fock_basis = Attribute(
    [
        "FockStateProjector",
        "NumberOperator",
        "TensorN",
        "Displacement",
        "Rotation",
        "Fourier",
        "SelectiveNumberArbitraryPhase",
        "Identity",
    ]
)

diagonal_in_position_basis = Attribute(
    [
        "QuadX",
        "TwoModeSum",
        "Identity",
    ]
)

attributes.composable_rotations.update(
    [
        "Displacement",
        "Squeezing",
        "TwoModeSqueezing",
        "Beamsplitter",
        "Rotation",
        "SelectiveQubitRotation",
        "SelectiveNumberArbitraryPhase",
        "TwoModeSum",
        "ConditionalRotation",
        "JaynesCummings",
        "AntiJaynesCummings",
        "ConditionalSqueezing",
        "ConditionalDisplacement",
        "ConditionalBeamsplitter",
        "ConditionalTwoModeSum",
    ]
)
