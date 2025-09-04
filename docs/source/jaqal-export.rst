Jaqal Export
============

This module provides functionality to export hybridlane/PennyLane quantum circuits to Jaqal (Just Another Quantum Assembly Language) format, which is used for trapped-ion quantum computers.

Basic Usage
-----------

.. code-block:: python

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

Working with Tapes
------------------

You can also export quantum tapes directly:

.. code-block:: python

    with qml.tape.QuantumTape() as tape:
        qml.RX(0.5, wires=0)
        qml.RY(1.0, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.expval(qml.PauliZ(0))

    jaqal_code = to_jaqal(tape)

Supported Operations
--------------------

Standard Quantum Gates
~~~~~~~~~~~~~~~~~~~~~~

The following standard quantum gates are fully supported and translated to native Jaqal operations:

- **Single-qubit gates**: RX, RY, RZ, Pauli-X/Y/Z, Hadamard
- **Two-qubit gates**: CNOT, CZ (decomposed using Mølmer-Sørensen gates)

PennyLane CV Operations
~~~~~~~~~~~~~~~~~~~~~~~

Standard PennyLane continuous variable operations generate placeholder comments:

- Displacement, Squeezing, Rotation
- Beamsplitter, TwoModeSqueezing
- Kerr, CrossKerr, CubicPhase
- InterferometerUnitary

Hybridlane-Specific Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hybridlane's CV and hybrid CV-DV operations generate placeholder comments indicating future support:

**CV Operations:**

- TwoModeSum, ModeSwap, Fourier
- FockStateProjector

**Hybrid CV-DV Operations:**

- JaynesCummings, AntiJaynesCummings
- ConditionalDisplacement, ConditionalBeamsplitter
- ConditionalRotation, ConditionalParity
- ConditionalTwoModeSqueezing, ConditionalTwoModeSum
- Rabi, SelectiveQubitRotation, SelectiveNumberArbitraryPhase

Note: Hybrid gates follow the convention [qumode_wire, qubit_wire] for 2-wire operations, or [qumode_wire(s), qubit_wire] for multi-wire operations.

Output Format
-------------

The generated Jaqal code follows the standard format:

.. code-block:: text

    // Generated from hybridlane circuit
    // Number of qubits: N

    register q[N]

    {
        prepare_all
        // Gate operations
        measure_all
    }

Gate Decompositions
-------------------

- **Hadamard**: Decomposed into Ry(π/2) and Rz(π)
- **CNOT**: Decomposed using Mølmer-Sørensen (MS) gates and single-qubit rotations
- **CZ**: Decomposed using MS gates and single-qubit rotations
- **Pauli gates**: Converted to rotations by π

Measurements
------------

Jaqal supports measurements in the computational basis. The export module handles:

- ``expval(PauliZ)`` measurements
- ``probs()`` for probability measurements
- ``sample()`` for shot-based measurements

For non-computational basis measurements, diagonalization would be required (future work).

Examples
--------

Bell State Preparation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @qml.qnode(qml.device("default.qubit", wires=2))
    def bell_state():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.probs(wires=[0, 1])
    
    jaqal_code = to_jaqal(bell_state)

GHZ State
~~~~~~~~~

.. code-block:: python

    @qml.qnode(qml.device("default.qubit", wires=3))
    def ghz_state():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.probs(wires=[0, 1, 2])
    
    jaqal_code = to_jaqal(ghz_state)

Limitations
-----------

1. **CV Operations**: Since Jaqal targets trapped-ion (discrete variable) systems, continuous variable operations are not natively supported and generate placeholder comments.

2. **Measurements**: Currently supports computational basis measurements. Non-computational basis measurements would require diagonalization circuits.

3. **Hybrid Operations**: Hybrid CV-DV operations are unique to hybridlane and not directly translatable to trapped-ion gates.

Future Work
-----------

- CV operation compilation strategies
- Measurement basis transformations
- Circuit optimization for trapped-ion architecture
- Integration with jaqalpaq via plugin system

References
----------

- `Jaqal Paper <https://arxiv.org/abs/2008.08042>`_
- `QSCOUT Platform <https://qscout.sandia.gov>`_
- `Jaqalpaq Repository <https://gitlab.com/jaqal/jaqalpaq>`_
