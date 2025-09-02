"""Export hybridlane circuits to Jaqal quantum assembly language."""

import pennylane as qml
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class JaqalGateType(Enum):
    """Standard Jaqal gate types."""
    PREPARE = "prepare_all"
    MEASURE = "measure_all"
    RX = "Rx"
    RY = "Ry"
    RZ = "Rz"
    MS = "MS"  # Molmer-Sorensen gate
    SXX = "Sxx"  # XX gate


@dataclass
class JaqalInstruction:
    """Represents a single Jaqal instruction."""
    gate_name: str
    qubits: List[int]
    parameters: List[float] = None
    
    def to_jaqal(self) -> str:
        """Convert to Jaqal syntax."""
        # Handle comment lines
        if self.gate_name.startswith("//"):
            return self.gate_name
        
        if self.parameters:
            params_str = " ".join(str(p) for p in self.parameters)
            qubits_str = " ".join(f"q[{q}]" for q in self.qubits)
            return f"{self.gate_name} {params_str} {qubits_str}"
        else:
            if self.qubits:
                qubits_str = " ".join(f"q[{q}]" for q in self.qubits)
                return f"{self.gate_name} {qubits_str}"
            else:
                return self.gate_name


class HybridlaneToJaqalTranslator:
    """Translates hybridlane/PennyLane circuits to Jaqal."""
    
    def __init__(self, num_qubits: Optional[int] = None):
        self.num_qubits = num_qubits
        self.instructions: List[JaqalInstruction] = []
        self.gate_map = self._initialize_gate_map()
    
    def _initialize_gate_map(self) -> Dict[str, callable]:
        """Initialize mapping from PennyLane gates to Jaqal instructions."""
        return {
            # Standard quantum gates (supported)
            "PauliX": self._pauli_x_to_jaqal,
            "PauliY": self._pauli_y_to_jaqal,
            "PauliZ": self._pauli_z_to_jaqal,
            "RX": self._rx_to_jaqal,
            "RY": self._ry_to_jaqal,
            "RZ": self._rz_to_jaqal,
            "Hadamard": self._hadamard_to_jaqal,
            "CNOT": self._cnot_to_jaqal,
            "CZ": self._cz_to_jaqal,
            
            # Hybridlane CV operations (placeholders - support coming soon)
            "FockStateProjector": self._cv_placeholder,
            "Fourier": self._cv_placeholder,
            "ModeSwap": self._cv_placeholder,
            "NumberOperator": self._cv_placeholder,
            "QuadOperator": self._cv_placeholder,
            "QuadP": self._cv_placeholder,
            "QuadX": self._cv_placeholder,
            "TwoModeSum": self._cv_placeholder,
            
            # Hybridlane hybrid CV-DV operations (placeholders - support coming soon)
            "AntiJaynesCummings": self._hybrid_placeholder,
            "ConditionalBeamsplitter": self._hybrid_placeholder,
            "ConditionalDisplacement": self._hybrid_placeholder,
            "ConditionalParity": self._hybrid_placeholder,
            "ConditionalRotation": self._hybrid_placeholder,
            "ConditionalTwoModeSqueezing": self._hybrid_placeholder,
            "ConditionalTwoModeSum": self._hybrid_placeholder,
            "Hybrid": self._hybrid_placeholder,
            "JaynesCummings": self._hybrid_placeholder,
            "Rabi": self._hybrid_placeholder,
            "SelectiveNumberArbitraryPhase": self._hybrid_placeholder,
            "SelectiveQubitRotation": self._hybrid_placeholder,
        }
    
    def _pauli_x_to_jaqal(self, op) -> List[JaqalInstruction]:
        """Convert Pauli-X to Jaqal Rx(pi)."""
        return [JaqalInstruction(JaqalGateType.RX.value, [op.wires[0]], [3.14159265359])]
    
    def _pauli_y_to_jaqal(self, op) -> List[JaqalInstruction]:
        """Convert Pauli-Y to Jaqal Ry(pi)."""
        return [JaqalInstruction(JaqalGateType.RY.value, [op.wires[0]], [3.14159265359])]
    
    def _pauli_z_to_jaqal(self, op) -> List[JaqalInstruction]:
        """Convert Pauli-Z to Jaqal Rz(pi)."""
        return [JaqalInstruction(JaqalGateType.RZ.value, [op.wires[0]], [3.14159265359])]
    
    def _rx_to_jaqal(self, op) -> List[JaqalInstruction]:
        """Convert RX rotation to Jaqal."""
        return [JaqalInstruction(JaqalGateType.RX.value, [op.wires[0]], [op.parameters[0]])]
    
    def _ry_to_jaqal(self, op) -> List[JaqalInstruction]:
        """Convert RY rotation to Jaqal."""
        return [JaqalInstruction(JaqalGateType.RY.value, [op.wires[0]], [op.parameters[0]])]
    
    def _rz_to_jaqal(self, op) -> List[JaqalInstruction]:
        """Convert RZ rotation to Jaqal."""
        return [JaqalInstruction(JaqalGateType.RZ.value, [op.wires[0]], [op.parameters[0]])]
    
    def _hadamard_to_jaqal(self, op) -> List[JaqalInstruction]:
        """Convert Hadamard to Jaqal using Ry and Rz rotations."""
        wire = op.wires[0]
        return [
            JaqalInstruction(JaqalGateType.RY.value, [wire], [1.5707963268]),  # pi/2
            JaqalInstruction(JaqalGateType.RZ.value, [wire], [3.14159265359]),  # pi
        ]
    
    def _cnot_to_jaqal(self, op) -> List[JaqalInstruction]:
        """Convert CNOT to Jaqal using MS gates and single-qubit rotations."""
        control, target = op.wires[0], op.wires[1]
        pi_2 = 1.5707963268
        pi = 3.14159265359
        
        return [
            JaqalInstruction(JaqalGateType.RY.value, [target], [pi_2]),
            JaqalInstruction(JaqalGateType.MS.value, [control, target], [0, pi_2]),
            JaqalInstruction(JaqalGateType.RX.value, [control], [-pi_2]),
            JaqalInstruction(JaqalGateType.RX.value, [target], [-pi_2]),
            JaqalInstruction(JaqalGateType.RY.value, [target], [-pi_2]),
        ]
    
    def _cz_to_jaqal(self, op) -> List[JaqalInstruction]:
        """Convert CZ to Jaqal using MS gates and single-qubit rotations."""
        q1, q2 = op.wires[0], op.wires[1]
        pi_2 = 1.5707963268
        pi = 3.14159265359
        
        return [
            JaqalInstruction(JaqalGateType.RY.value, [q2], [pi_2]),
            JaqalInstruction(JaqalGateType.MS.value, [q1, q2], [0, pi_2]),
            JaqalInstruction(JaqalGateType.RX.value, [q1], [-pi_2]),
            JaqalInstruction(JaqalGateType.RX.value, [q2], [-pi_2]),
            JaqalInstruction(JaqalGateType.RY.value, [q2], [-pi_2]),
            JaqalInstruction(JaqalGateType.RZ.value, [q2], [pi]),
        ]
    
    def _cv_placeholder(self, op) -> List[JaqalInstruction]:
        """Placeholder for continuous variable operations.
        
        CV operations are not natively supported in Jaqal (trapped-ion architecture).
        Support for CV operation emulation/compilation coming soon.
        """
        op_name = op.name
        wires = list(op.wires)
        # Return a comment instruction placeholder
        comment = f"// CV Operation: {op_name} on modes {wires} - Support coming soon"
        return [JaqalInstruction(comment, [], [])]
    
    def _hybrid_placeholder(self, op) -> List[JaqalInstruction]:
        """Placeholder for hybrid CV-DV operations.
        
        Hybrid CV-DV operations are not natively supported in Jaqal (trapped-ion architecture).
        Support for hybrid operation decomposition/compilation coming soon.
        """
        op_name = op.name
        wires = list(op.wires)
        # Return a comment instruction placeholder
        comment = f"// Hybrid CV-DV Operation: {op_name} on wires/modes {wires} - Support coming soon"
        return [JaqalInstruction(comment, [], [])]
    
    def translate_operation(self, op):
        """Translate a single PennyLane operation to Jaqal instructions."""
        op_name = op.name
        
        if op_name in self.gate_map:
            return self.gate_map[op_name](op)
        else:
            raise NotImplementedError(f"Gate '{op_name}' not yet supported for Jaqal export")
    
    def translate_circuit(self, tape: qml.tape.QuantumTape) -> str:
        """Translate a PennyLane tape to Jaqal program."""
        self.instructions = []
        
        # Determine number of qubits
        if self.num_qubits is None:
            self.num_qubits = len(tape.wires)
        
        # Add preparation
        self.instructions.append(JaqalInstruction(JaqalGateType.PREPARE.value, [], []))
        
        # Translate operations
        for op in tape.operations:
            jaqal_ops = self.translate_operation(op)
            self.instructions.extend(jaqal_ops)
        
        # Add measurements if needed
        if tape.measurements:
            self.instructions.append(JaqalInstruction(JaqalGateType.MEASURE.value, [], []))
        
        # Generate Jaqal code
        return self.generate_jaqal_code()
    
    def generate_jaqal_code(self) -> str:
        """Generate the final Jaqal code."""
        lines = []
        
        # Header
        lines.append(f"// Generated from hybridlane circuit")
        lines.append(f"// Number of qubits: {self.num_qubits}")
        lines.append("")
        
        # Register declaration
        lines.append(f"register q[{self.num_qubits}]")
        lines.append("")
        
        # Main circuit block
        lines.append("{")
        for instruction in self.instructions:
            lines.append(f"    {instruction.to_jaqal()}")
        lines.append("}")
        
        return "\n".join(lines)


def to_jaqal(qnode_or_tape, *args, **kwargs) -> str:
    """
    Export a hybridlane/PennyLane circuit to Jaqal format.
    
    Args:
        qnode_or_tape: Either a QNode or a QuantumTape to export
        *args: Arguments to pass to QNode if applicable
        **kwargs: Keyword arguments to pass to QNode if applicable
    
    Returns:
        str: Jaqal program as a string
    """
    # Handle QNode
    if isinstance(qnode_or_tape, qml.QNode):
        # Execute the QNode to get the tape
        qnode_or_tape(*args, **kwargs)
        tape = qnode_or_tape._tape
    elif isinstance(qnode_or_tape, qml.tape.QuantumTape):
        tape = qnode_or_tape
    else:
        raise TypeError("Input must be a QNode or QuantumTape")
    
    # Translate to Jaqal
    translator = HybridlaneToJaqalTranslator()
    return translator.translate_circuit(tape)