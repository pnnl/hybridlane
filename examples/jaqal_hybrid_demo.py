#!/usr/bin/env python
"""
Demo showing hybridlane-specific CV and hybrid CV-DV operations in Jaqal export.

Since Jaqal is designed for trapped-ion (discrete variable) systems, continuous
variable and hybrid operations generate placeholder comments indicating future support.
"""

import pennylane as qml
import hybridlane as hqml
from hybridlane.export import to_jaqal


def hybrid_cv_dv_example():
    """Demonstrate a hybrid CV-DV circuit with placeholders."""
    print("=" * 60)
    print("Hybrid CV-DV Circuit Example")
    print("=" * 60)
    print("\nThis circuit mixes standard DV gates with CV and hybrid operations.")
    print("CV and hybrid operations will show as placeholders in Jaqal.\n")
    print("Wire convention: wires 0-1 are qumodes (cavities), wires 2-3 are qubits\n")
    
    # Use a tape to avoid device restrictions
    with qml.tape.QuantumTape() as tape:
        # Standard quantum gates on qubits (fully supported)
        qml.Hadamard(wires=2)
        qml.RY(0.5, wires=3)
        
        # CV operations on qumodes (placeholders)
        hqml.TwoModeSum(1.0, wires=[0, 1])  # CV operation on modes 0,1
        hqml.ModeSwap(wires=[0, 1])  # Swap modes 0,1
        hqml.Fourier(wires=0)  # Fourier on mode 0
        
        # Hybrid CV-DV operations (placeholders)
        # Note: Hybrid gates have convention [qumode_wire, qubit_wire]
        hqml.JaynesCummings(0.8, 0.3, wires=[0, 2])  # Mode 0 interacts with qubit 2
        hqml.ConditionalDisplacement(1.5, 0.2, wires=[1, 3])  # Mode 1 with qubit 3 control
        hqml.Rabi(2.0, wires=[0, 3])  # Mode 0 interacts with qubit 3
        
        # More standard gates on qubits
        qml.CNOT(wires=[2, 3])
        qml.CZ(wires=[3, 2])
        
        # Additional hybrid operations
        hqml.ConditionalRotation(0.7, wires=[0, 2])  # Mode 0 with qubit 2 control
        hqml.ConditionalBeamsplitter(0.5, 0.2, wires=[0, 1, 3])  # Modes 0,1 with qubit 3 control
        
        # Measure only the qubits (Jaqal only supports qubit measurements)
        qml.probs(wires=[2, 3])
    
    # Export to Jaqal
    jaqal_code = to_jaqal(tape)
    print(jaqal_code)
    print()


def selective_operations_example():
    """Demonstrate selective and conditional operations."""
    print("=" * 60)
    print("Selective Operations Example")
    print("=" * 60)
    print("\nThese operations perform qubit rotations based on cavity state.")
    print("Wire convention: wires 0-2 are qumodes, wire 3 is a qubit\n")
    
    with qml.tape.QuantumTape() as tape:
        # Prepare initial qubit state
        qml.RX(0.3, wires=3)  # Rotate qubit 3
        
        # Selective operations: qubit rotation based on qumode Fock state
        # Format: [qumode_wire, qubit_wire]
        hqml.SelectiveQubitRotation(0.5, 0.2, n=1, wires=[0, 3])  # Rotate qubit 3 if mode 0 in n=1
        hqml.SelectiveNumberArbitraryPhase(1.0, n=2, wires=[1, 3])  # Phase qubit 3 if mode 1 in n=2
        
        # Anti-Jaynes-Cummings (Blue sideband) - mode 0 with qubit 3
        hqml.AntiJaynesCummings(0.4, 0.1, wires=[0, 3])
        
        # Conditional operations
        hqml.ConditionalParity(wires=[0, 3])  # Mode 0 with qubit 3 control
        hqml.ConditionalTwoModeSum(0.8, wires=[0, 1, 3])  # Modes 0,1 with qubit 3 control
        hqml.ConditionalTwoModeSqueezing(0.3, 0.1, wires=[1, 2, 3])  # Modes 1,2 with qubit 3 control
        
        # Measure the qubit
        qml.expval(qml.PauliZ(3))
    
    jaqal_code = to_jaqal(tape)
    print(jaqal_code)
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Hybridlane to Jaqal Export - CV/Hybrid Operations Demo")
    print("=" * 60)
    print("\nNOTE: Jaqal is designed for trapped-ion quantum computers which")
    print("are discrete variable (DV) systems. Continuous variable (CV) and")
    print("hybrid CV-DV operations are shown as placeholders with comments")
    print("indicating that support is coming soon.\n")
    
    hybrid_cv_dv_example()
    selective_operations_example()
    
    print("=" * 60)
    print("Export complete!")
    print("\nThe placeholder comments indicate where future compilation")
    print("strategies will be implemented to decompose CV and hybrid")
    print("operations into trapped-ion native gates.")
    print("=" * 60)


if __name__ == "__main__":
    main()