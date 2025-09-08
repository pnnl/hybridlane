# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import pennylane as qml
from pennylane.operation import Operator
from pennylane.ops.cv import CVOperation

import hybridlane as hqml

from ...ops import Hybrid

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
    qml.Beamsplitter: "cv_bs",
    qml.ControlledAddition: None,
    qml.ControlledPhase: None,
    qml.CrossKerr: None,
    qml.CubicPhase: None,
    qml.Displacement: "cv_d",
    qml.InterferometerUnitary: None,
    qml.Kerr: None,
    qml.QuadraticPhase: None,
    qml.Rotation: "cv_r",
    qml.Squeezing: "cv_sq",
    qml.TwoModeSqueezing: "cv_sq2",
    hqml.TwoModeSum: "cv_sum",
    hqml.ModeSwap: None,  # has decomposition in terms of beamsplitter
    hqml.Fourier: None,  # has decomposition in terms of Rotation
}

# Finally, the hybrid gates in our library -> bosonic qiskit
# Each of these gates has both qumodes and qubits
#
#  [1] SQR is marked as "todo" in bosonic qiskit:
#      https://github.com/C2QA/bosonic-qiskit/blob/52a1a7ffe4a4c7b06b5828f8956d905e0d9d662a/c2qa/circuit.py#L692C6-L693C21
#
hybrid_gate_map: dict[type[Hybrid], str | None] = {
    hqml.ConditionalRotation: "cv_c_r",
    hqml.ConditionalParity: None,  # special case of conditional rotation
    hqml.SelectiveQubitRotation: None,
    hqml.SelectiveNumberArbitraryPhase: "cv_snap",
    hqml.JaynesCummings: "cv_jc",
    hqml.AntiJaynesCummings: "cv_ajc",
    hqml.Rabi: "cv_rb",  # todo: verify this gate in particular
    hqml.ConditionalDisplacement: "cv_c_d",
    hqml.ConditionalBeamsplitter: "cv_c_bs",
    hqml.ConditionalTwoModeSqueezing: None,
    hqml.ConditionalTwoModeSum: "cv_c_sum",
}

misc_gate_map = {qml.Barrier: "barrier"}

supported_operations = set(
    k
    for k, v in (dv_gate_map | cv_gate_map | hybrid_gate_map | misc_gate_map).items()
    if v is not None
)
