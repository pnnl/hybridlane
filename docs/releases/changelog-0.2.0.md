# Release 0.2.0

### New features

#### OpenQASM exporting

- Adds the function `hqml.to_openqasm()` that exports circuits to a superset of OpenQASM 3.0, with an optional flag to enforce compliance with the OpenQASM standard. (#13)
  - The modifications to OpenQASM are detailed in the documentation. We also include a standard library in `examples/cvstdgates.inc`.

#### Conversions from PennyLane

- Adds our own versions of common existing CV Pennylane gates like Displacement, Squeezing, and Beamsplitter. (#13)
  - Each of these gates matches the convention in Liu et al. (arXiv:2407.10381) and serves as the definition of the CV OpenQASM library.
- Introduces the `from_pennylane` circuit transform that converts Pennylane gates and measurements into Hybridlane ones. (#13)
  - We hope this reduces confusion arising from different definitions of gates (e.g. `qml.Rotation` and `hqml.Rotation` differ by a minus sign) and measurements (e.g. `qml.expval` vs `hqml.expval`).
- Updates `hybrid.bosonicqiskit` device to use `from_pennylane` in its preprocessing step, simplifying the implementation. (#13)

### Improvements

- Adds truncated gate labels to improve circuit drawing. (#13)

### Breaking changes

- Redefines the wire order in hybrid gates to put qubits before qumodes: `(*qubits, *qumodes)`. (#13)
  - This simplified the implementation of the `to_openqasm` function by making the gates consistent with OpenQASM's `ctrl` modifier.
- Updates the `Rabi` gate to properly use a complex parameter. (#13)

### Fixes

- Fixes the math in `simplify`, `pow`, and/or `adjoint` methods for the `ConditionalTwoModeSqueezing` and `Rabi` gates. (#13)

### Other changes

- Drops the intersphinx mappings for numpy, scipy, and qiskit.

### Contributors

This release contains contributions from:
Jim Furches
