# Hybridlane Jaqal Export

This module provides functionality to export hybridlane/PennyLane quantum circuits to Jaqal (Just Another Quantum Assembly Language) format, which is used for trapped-ion quantum computers.

## Features

### Supported Operations

The following standard quantum gates are fully supported and translated to native Jaqal operations:

- **Single-qubit gates**: RX, RY, RZ, Pauli-X/Y/Z, Hadamard
- **Two-qubit gates**: CNOT, CZ (decomposed using Mølmer-Sørensen gates)

### Placeholder Support for Hybridlane Operations

Since Jaqal is designed for trapped-ion (discrete variable) systems, the following hybridlane-specific operations generate placeholder comments indicating future support:

#### Continuous Variable (CV) Operations
- `TwoModeSum`
- `ModeSwap`
- `Fourier`
- `FockStateProjector`
- `NumberOperator`
- `QuadOperator`, `QuadX`, `QuadP`

#### Hybrid CV-DV Operations
- `JaynesCummings` / `AntiJaynesCummings`
- `ConditionalDisplacement`
- `ConditionalBeamsplitter`
- `ConditionalRotation`
- `ConditionalParity`
- `ConditionalTwoModeSqueezing`
- `ConditionalTwoModeSum`
- `Rabi`
- `SelectiveQubitRotation`
- `SelectiveNumberArbitraryPhase`

## Usage

### Basic Example

```python
import pennylane as qml
from hybridlane.export import to_jaqal

# Create a quantum circuit
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def bell_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

# Export to Jaqal
jaqal_code = to_jaqal(bell_circuit)
print(jaqal_code)
```

### Working with Tapes

You can also export quantum tapes directly:

```python
with qml.tape.QuantumTape() as tape:
    qml.RX(0.5, wires=0)
    qml.RY(1.0, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.expval(qml.PauliZ(0))

jaqal_code = to_jaqal(tape)
```

### Hybrid CV-DV Circuits

For circuits containing hybridlane-specific operations:

```python
import hybridlane as hqml

with qml.tape.QuantumTape() as tape:
    # Standard gates (fully supported)
    qml.Hadamard(wires=0)
    
    # CV operation (placeholder)
    hqml.TwoModeSum(1.0, wires=[1, 2])
    
    # Hybrid operation (placeholder)
    hqml.JaynesCummings(0.8, 0.3, wires=[0, 1])
    
    qml.probs(wires=[0, 1])

jaqal_code = to_jaqal(tape)
# CV and hybrid operations will appear as comments in the output
```

## Output Format

The generated Jaqal code follows the standard format:

```jaqal
// Generated from hybridlane circuit
// Number of qubits: N

register q[N]

{
    prepare_all
    // Gate operations
    measure_all
}
```

## Gate Decompositions

- **Hadamard**: Decomposed into Ry(π/2) and Rz(π)
- **CNOT**: Decomposed using Mølmer-Sørensen (MS) gates and single-qubit rotations
- **CZ**: Decomposed using MS gates and single-qubit rotations
- **Pauli gates**: Converted to rotations by π

## Future Work

The placeholder system for CV and hybrid operations provides a clear roadmap for future development:

1. **CV Operation Compilation**: Develop strategies to compile continuous variable operations into trapped-ion native gates
2. **Hybrid Operation Decomposition**: Implement decomposition schemes for hybrid CV-DV operations
3. **Optimization**: Add circuit optimization passes specific to trapped-ion architectures
4. **Error Handling**: Implement more sophisticated error handling for unsupported operations

## Testing

Run the test suite:

```bash
pytest test/export/test_jaqal.py
pytest test/export/test_hybridlane_ops_jaqal.py
```

## Examples

See the `examples/` directory for complete examples:
- `jaqal_export_demo.py`: Basic export functionality
- `jaqal_hybrid_demo.py`: Hybrid CV-DV circuits with placeholders