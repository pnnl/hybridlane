#!/usr/bin/env python
"""
Demo script showing how to export hybridlane/PennyLane circuits to Jaqal format.
"""

import pennylane as qml
from hybridlane.export import to_jaqal


def bell_state_example():
    """Create and export a Bell state preparation circuit."""
    print("=" * 60)
    print("Bell State Preparation Circuit")
    print("=" * 60)
    
    # Create a simple Bell state circuit
    dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(dev)
    def bell_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])
    
    # Export to Jaqal
    jaqal_code = to_jaqal(bell_circuit)
    print(jaqal_code)
    print()


def ghz_state_example():
    """Create and export a GHZ state preparation circuit."""
    print("=" * 60)
    print("GHZ State Preparation Circuit")
    print("=" * 60)
    
    # Create a GHZ state circuit
    dev = qml.device("default.qubit", wires=3)
    
    @qml.qnode(dev)
    def ghz_circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.probs(wires=[0, 1, 2])
    
    # Export to Jaqal
    jaqal_code = to_jaqal(ghz_circuit)
    print(jaqal_code)
    print()


def rotation_example():
    """Create and export a circuit with rotation gates."""
    print("=" * 60)
    print("Rotation Gates Example")
    print("=" * 60)
    
    # Create a circuit with various rotations
    dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(dev)
    def rotation_circuit():
        qml.RX(0.5, wires=0)
        qml.RY(1.0, wires=0)
        qml.RZ(1.5, wires=0)
        qml.RX(0.7, wires=1)
        qml.CZ(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    
    # Export to Jaqal
    jaqal_code = to_jaqal(rotation_circuit)
    print(jaqal_code)
    print()


def main():
    """Run all examples."""
    print("\nHybridlane to Jaqal Export Examples\n")
    
    bell_state_example()
    ghz_state_example()
    rotation_example()
    
    print("=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()