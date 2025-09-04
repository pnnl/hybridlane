"""Tests for Jaqal export with hybridlane-specific operations."""

import pennylane as qml
import pytest
import hybridlane as hqml
from hybridlane.export import to_jaqal


class TestHybridlaneOpsJaqalExport:
    """Test suite for exporting hybridlane-specific operations to Jaqal."""
    
    def test_cv_operations_placeholder(self):
        """Test that CV operations generate appropriate placeholder comments."""
        # Use a tape directly to avoid device restrictions
        with qml.tape.QuantumTape() as tape:
            # Mix of standard and CV operations
            qml.RX(0.5, wires=0)
            hqml.TwoModeSum(1.0, wires=[1, 2])  # CV operation
            hqml.ModeSwap(wires=[2, 3])  # CV operation
            hqml.Fourier(wires=1)  # CV operation
            qml.expval(qml.PauliZ(0))
        
        jaqal_code = to_jaqal(tape)
        
        # Check for standard gate
        assert "Rx 0.5 q[0]" in jaqal_code
        # Check for CV operation placeholders
        assert "// CV Operation: TwoModeSum" in jaqal_code
        assert "// CV Operation: ModeSwap" in jaqal_code
        assert "// CV Operation: Fourier" in jaqal_code
        assert "Support coming soon" in jaqal_code
    
    def test_hybrid_operations_placeholder(self):
        """Test that hybrid CV-DV operations generate appropriate placeholder comments."""
        # Use a tape directly to avoid device restrictions
        with qml.tape.QuantumTape() as tape:
            # Mix of standard and hybrid operations
            qml.Hadamard(wires=0)
            hqml.ConditionalDisplacement(1.0, 0, wires=[1, 0])  # 2 wires: mode 1, qubit 0
            hqml.JaynesCummings(0.5, 0.3, wires=[2, 1])  # 2 wires: mode 2, qubit 1; 2 params
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=[0, 1])
        
        jaqal_code = to_jaqal(tape)
        
        # Check for standard gates (Hadamard decomposition)
        assert "Ry" in jaqal_code
        assert "Rz" in jaqal_code
        # Check for CNOT decomposition
        assert "MS" in jaqal_code
        # Check for hybrid operation placeholders
        assert "// Hybrid CV-DV Operation: ConditionalDisplacement" in jaqal_code
        assert "// Hybrid CV-DV Operation: JaynesCummings" in jaqal_code
        assert "Support coming soon" in jaqal_code
    
    def test_mixed_circuit(self):
        """Test a circuit mixing standard, CV, and hybrid operations."""
        # Use a tape directly to avoid device restrictions
        with qml.tape.QuantumTape() as tape:
            # Standard gates
            qml.RY(0.3, wires=0)
            qml.PauliX(wires=1)
            
            # CV operations
            hqml.TwoModeSum(0.5, wires=[2, 3])
            hqml.ModeSwap(wires=[2, 3])
            hqml.Fourier(wires=2)
            
            # Hybrid operations
            hqml.Rabi(1.5, wires=[2, 0])  # 2 wires: mode 2, qubit 0
            hqml.ConditionalRotation(0.7, wires=[2, 1])  # 2 wires: mode 2, qubit 1; 1 param (theta)
            
            # More standard gates
            qml.CZ(wires=[0, 1])
            
            qml.state()
        
        jaqal_code = to_jaqal(tape)
        
        # Verify the circuit has all types of operations
        assert "Ry 0.3 q[0]" in jaqal_code
        assert "Rx 3.14159" in jaqal_code  # PauliX
        assert "// CV Operation: TwoModeSum" in jaqal_code
        assert "// CV Operation: ModeSwap" in jaqal_code
        assert "// CV Operation: Fourier" in jaqal_code
        assert "// Hybrid CV-DV Operation: Rabi" in jaqal_code
        assert "// Hybrid CV-DV Operation: ConditionalRotation" in jaqal_code
        assert "MS" in jaqal_code  # From CZ decomposition
    
    def test_selective_operations(self):
        """Test selective qubit/number operations."""
        # Use a tape directly to avoid device restrictions
        with qml.tape.QuantumTape() as tape:
            hqml.SelectiveQubitRotation(0.5, 0.2, 1, wires=[0, 1])  # theta, phi, n, wires: mode 0, qubit 1
            hqml.SelectiveNumberArbitraryPhase(1.0, 2, wires=[2, 1])  # phi, n, wires: mode 2, qubit 1
            hqml.AntiJaynesCummings(0.3, 0.2, wires=[2, 0])  # theta, phi, wires: mode 2, qubit 0
            qml.expval(qml.PauliZ(0))
        
        jaqal_code = to_jaqal(tape)
        
        assert "// Hybrid CV-DV Operation: SelectiveQubitRotation" in jaqal_code
        assert "// Hybrid CV-DV Operation: SelectiveNumberArbitraryPhase" in jaqal_code
        assert "// Hybrid CV-DV Operation: AntiJaynesCummings" in jaqal_code
        assert "Support coming soon" in jaqal_code
    
    def test_conditional_operations(self):
        """Test various conditional operations."""
        # Use a tape directly to avoid device restrictions
        with qml.tape.QuantumTape() as tape:
            hqml.ConditionalBeamsplitter(0.5, 0.2, wires=[1, 2, 0])  # theta, phi, wires (3 wires: modes 1,2, qubit 0)
            hqml.ConditionalTwoModeSqueezing(0.3, 0.1, wires=[2, 3, 0])  # r, phi, wires (3 wires: modes 2,3, qubit 0)
            hqml.ConditionalTwoModeSum(0.7, wires=[1, 2, 0])  # lambda, wires (3 wires: modes 1,2, qubit 0)
            hqml.ConditionalParity(wires=[1, 0])  # wires (2 wires: mode 1, qubit 0)
            qml.probs(wires=[0, 1, 2, 3])
        
        jaqal_code = to_jaqal(tape)
        
        assert "// Hybrid CV-DV Operation: ConditionalBeamsplitter" in jaqal_code
        assert "// Hybrid CV-DV Operation: ConditionalTwoModeSqueezing" in jaqal_code
        assert "// Hybrid CV-DV Operation: ConditionalTwoModeSum" in jaqal_code
        assert "// Hybrid CV-DV Operation: ConditionalParity" in jaqal_code
        assert "Support coming soon" in jaqal_code