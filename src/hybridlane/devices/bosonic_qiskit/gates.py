# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

# For entries that list `None`, they are listed for completeness. We should force the user to compile
# their circuit to the basis defined by gates that have methods listed. However, some of these gates
# don't have decompositions, which will be an issue.

# This is a mapping from the pennylane class -> qiskit method name
dv_gate_map: dict[str, str] = {
    "Identity": "id",
    "Hadamard": "h",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "S": "s",
    "Adjoint(S)": "sdg",
    "T": "t",
    "Adjoint(T)": "tdg",
    "SX": "sx",
    "CNOT": "cx",
    "CZ": "cz",
    "CY": "cy",
    "CH": "ch",
    "SWAP": "swap",
    "ISWAP": "iswap",
    "ECR": "ecr",
    "CSWAP": "cswap",
    "Toffoli": "ccx",
    "Rot": "u",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "PhaseShift": "p",
    "ControlledPhaseShift": "cp",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "IsingXX": "rxx",
    "IsingYY": "ryy",
    "IsingZZ": "rzz",
}

# This map is CV operators of pennylane and our library -> bosonic qiskit
# Everything here only acts on qumodes
cv_gate_map: dict[str, str | None] = {
    "Beamsplitter": "cv_bs",
    "CubicPhase": None,
    "Displacement": "cv_d",
    "Fourier": None,  # has decomposition in terms of Rotation
    "Kerr": None,
    "ModeSwap": None,  # has decomposition in terms of beamsplitter
    "Rotation": "cv_r",
    "Squeezing": "cv_sq",
    "TwoModeSqueezing": "cv_sq2",
    "TwoModeSum": "cv_sum",
}

# Finally, the hybrid gates in our library -> bosonic qiskit
# Each of these gates has both qumodes and qubits
hybrid_gate_map: dict[str, str | None] = {
    "AntiJaynesCummings": "cv_ajc",
    "ConditionalBeamsplitter": "cv_c_bs",
    "ConditionalDisplacement": "cv_c_d",
    "ConditionalParity": None,  # special case of conditional rotation
    "ConditionalRotation": "cv_c_r",
    "ConditionalSqueezing": "cv_c_sq",
    "ConditionalTwoModeSqueezing": None,
    "ConditionalTwoModeSum": "cv_c_sum",
    "JaynesCummings": "cv_jc",
    "Rabi": "cv_rb",
    "SelectiveNumberArbitraryPhase": "cv_snap",
    "SelectiveQubitRotation": "cv_sqr",
}

misc_gate_map = {"Barrier": "barrier"}

supported_operations = set(
    k
    for k, v in (dv_gate_map | cv_gate_map | hybrid_gate_map | misc_gate_map).items()
    if v is not None
)
