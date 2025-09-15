# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.cv import CVOperation

import hybridlane as hqml

from ...ops.mixins import Hybrid

# For entries that list `None`, they are listed for completeness. We should force the user to compile
# their circuit to the basis defined by gates that have methods listed. However, some of these gates
# don't have decompositions, which will be an issue.

# This is a mapping from the pennylane class -> qiskit method name
dv_gate_map: dict[type[Operator], str] = {
    # Todo: How do we handle sdg and tdg? Qiskit has the method, but I'm not sure how pennylane handles that,
    # or if they just wrap it in e.g. Adjoint(S())
    qml.Identity: "id",
    qml.Hadamard: "h",
    qml.PauliX: "x",
    qml.PauliY: "y",
    qml.PauliZ: "z",
    qml.S: "s",
    qml.T: "t",
    qml.SX: "sx",
    qml.CNOT: "cx",
    qml.CZ: "cz",
    qml.CY: "cy",
    qml.CH: "ch",
    qml.SWAP: "swap",
    qml.ISWAP: "iswap",
    qml.ECR: "ecr",
    qml.CSWAP: "cswap",
    qml.Toffoli: "ccx",
    qml.Rot: "u",
    qml.RX: "rx",
    qml.RY: "ry",
    qml.RZ: "rz",
    qml.PhaseShift: "p",
    qml.ControlledPhaseShift: "cp",
    qml.CRX: "crx",
    qml.CRY: "cry",
    qml.CRZ: "crz",
    qml.IsingXX: "rxx",
    qml.IsingYY: "ryy",
    qml.IsingZZ: "rzz",
}

# This map is CV operators of pennylane and our library -> bosonic qiskit
# Everything here only acts on qumodes
cv_gate_map: dict[type[CVOperation], str | None] = {
    hqml.Beamsplitter: "cv_bs",
    hqml.CubicPhase: None,
    hqml.Displacement: "cv_d",
    hqml.Fourier: None,  # has decomposition in terms of Rotation
    hqml.Kerr: None,
    hqml.ModeSwap: None,  # has decomposition in terms of beamsplitter
    hqml.Rotation: "cv_r",
    hqml.Squeezing: "cv_sq",
    hqml.TwoModeSqueezing: "cv_sq2",
    hqml.TwoModeSum: "cv_sum",
}

# Finally, the hybrid gates in our library -> bosonic qiskit
# Each of these gates has both qumodes and qubits
#
#  [1] SQR is marked as "todo" in bosonic qiskit:
#      https://github.com/C2QA/bosonic-qiskit/blob/52a1a7ffe4a4c7b06b5828f8956d905e0d9d662a/c2qa/circuit.py#L692C6-L693C21
#
hybrid_gate_map: dict[type[Hybrid], str | None] = {
    hqml.AntiJaynesCummings: "cv_ajc",
    hqml.ConditionalBeamsplitter: "cv_c_bs",
    hqml.ConditionalDisplacement: "cv_c_d",
    hqml.ConditionalParity: None,  # special case of conditional rotation
    hqml.ConditionalRotation: "cv_c_r",
    hqml.ConditionalSqueezing: None,
    hqml.ConditionalTwoModeSqueezing: None,
    hqml.ConditionalTwoModeSum: "cv_c_sum",
    hqml.JaynesCummings: "cv_jc",
    hqml.Rabi: "cv_rb",
    hqml.SelectiveNumberArbitraryPhase: "cv_snap",
    hqml.SelectiveQubitRotation: None,
}

misc_gate_map = {qml.Barrier: "barrier"}

supported_operations = set(
    k
    for k, v in (dv_gate_map | cv_gate_map | hybrid_gate_map | misc_gate_map).items()
    if v is not None
)
