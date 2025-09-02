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
    
    # Use a tape to avoid device restrictions
    with qml.tape.QuantumTape() as tape:
        # Standard quantum gates (fully supported)
        qml.Hadamard(wires=0)
        qml.RY(0.5, wires=1)
        
        # CV operations (placeholders)
        hqml.TwoModeSum(1.0, wires=[2, 3])
        hqml.ModeSwap(wires=[3, 4])
        hqml.Fourier(wires=2)
        
        # Hybrid CV-DV operations (placeholders)
        hqml.JaynesCummings(0.8, 0.3, wires=[0, 2])  # Qubit-cavity interaction
        hqml.ConditionalDisplacement(1.5, 0.2, wires=[3, 1])  # Conditional displacement
        hqml.Rabi(2.0, wires=[1, 4])  # Rabi interaction
        
        # More standard gates
        qml.CNOT(wires=[0, 1])
        qml.CZ(wires=[1, 0])
        
        # Conditional operations (placeholders)
        hqml.ConditionalRotation(0.7, wires=[0, 2])
        hqml.ConditionalBeamsplitter(0.5, 0.2, wires=[1, 3, 4])
        
        qml.probs(wires=[0, 1])
    
    # Export to Jaqal
    jaqal_code = to_jaqal(tape)
    print(jaqal_code)
    print()


def selective_operations_example():
    """Demonstrate selective and conditional operations."""
    print("=" * 60)
    print("Selective Operations Example")
    print("=" * 60)
    print("\nThese operations perform qubit rotations based on cavity state.\n")
    
    with qml.tape.QuantumTape() as tape:
        # Prepare initial state
        qml.RX(0.3, wires=0)
        
        # Selective operations based on Fock state
        hqml.SelectiveQubitRotation(0.5, 0.2, n=1, wires=[0, 1])  # Rotate if n=1
        hqml.SelectiveNumberArbitraryPhase(1.0, n=2, wires=[0, 1])  # Phase if n=2
        
        # Anti-Jaynes-Cummings (Blue sideband)
        hqml.AntiJaynesCummings(0.4, 0.1, wires=[0, 1])
        
        # Conditional operations
        hqml.ConditionalParity(wires=[0, 1])
        hqml.ConditionalTwoModeSum(0.8, wires=[0, 1, 2])
        hqml.ConditionalTwoModeSqueezing(0.3, 0.1, wires=[0, 1, 2])
        
        qml.expval(qml.PauliZ(0))
    
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