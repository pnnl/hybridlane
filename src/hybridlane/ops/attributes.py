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
