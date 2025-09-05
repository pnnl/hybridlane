"""Tests for Jaqal export functionality."""

import pennylane as qml
import pytest
from hybridlane.export import to_jaqal


class TestJaqalExport:
    """Test suite for Jaqal export functionality."""
    
    def test_simple_circuit(self):
        """Test export of a simple single-qubit circuit."""
        dev = qml.device("default.qubit", wires=1)
        
        @qml.qnode(dev)
        def circuit():
            qml.RX(0.5, wires=0)
            qml.RY(1.0, wires=0)
            return qml.expval(qml.PauliZ(0))
        
        jaqal_code = to_jaqal(circuit)
        
        assert "register q[1]" in jaqal_code
        assert "prepare_all" in jaqal_code
        assert "Rx 0.5 q[0]" in jaqal_code
        assert "Ry 1.0 q[0]" in jaqal_code
        assert "measure_all" in jaqal_code
    
    def test_two_qubit_circuit(self):
        """Test export of a two-qubit circuit with entanglement."""
        dev = qml.device("default.qubit", wires=2)
        
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        
        jaqal_code = to_jaqal(circuit)
        
        assert "register q[2]" in jaqal_code
        assert "prepare_all" in jaqal_code
        # Hadamard decomposition
        assert "Ry" in jaqal_code
        assert "Rz" in jaqal_code
        # CNOT decomposition using MS gates
        assert "MS" in jaqal_code
        assert "measure_all" in jaqal_code
    
    def test_pauli_gates(self):
        """Test conversion of Pauli gates."""
        dev = qml.device("default.qubit", wires=1)
        
        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            qml.PauliY(wires=0)
            qml.PauliZ(wires=0)
            return qml.expval(qml.PauliZ(0))
        
        jaqal_code = to_jaqal(circuit)
        
        # Pauli gates should be converted to rotations by pi
        assert "Rx 3.14159" in jaqal_code
        assert "Ry 3.14159" in jaqal_code
        assert "Rz 3.14159" in jaqal_code
    
    def test_multi_qubit_circuit(self):
        """Test export of a multi-qubit circuit."""
        dev = qml.device("default.qubit", wires=3)
        
        @qml.qnode(dev)
        def circuit():
            qml.RX(0.1, wires=0)
            qml.RY(0.2, wires=1)
            qml.RZ(0.3, wires=2)
            qml.CNOT(wires=[0, 1])
            qml.CZ(wires=[1, 2])
            return qml.probs(wires=[0, 1, 2])
        
        jaqal_code = to_jaqal(circuit)
        
        assert "register q[3]" in jaqal_code
        assert "Rx 0.1 q[0]" in jaqal_code
        assert "Ry 0.2 q[1]" in jaqal_code
        assert "Rz 0.3 q[2]" in jaqal_code
    
    def test_tape_export(self):
        """Test direct export from a QuantumTape."""
        with qml.tape.QuantumTape() as tape:
            qml.RX(1.5, wires=0)
            qml.RY(2.0, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
        
        jaqal_code = to_jaqal(tape)
        
        assert "register q[2]" in jaqal_code
        assert "Rx 1.5 q[0]" in jaqal_code
        assert "Ry 2.0 q[1]" in jaqal_code
    
    def test_unsupported_gate_error(self):
        """Test that unsupported gates raise appropriate errors."""
        dev = qml.device("default.qubit", wires=3)
        
        @qml.qnode(dev)
        def circuit():
            qml.Toffoli(wires=[0, 1, 2])
            return qml.expval(qml.PauliZ(0))
        
        with pytest.raises(NotImplementedError, match="not yet supported"):
            to_jaqal(circuit)