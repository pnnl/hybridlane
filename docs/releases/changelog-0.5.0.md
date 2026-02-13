# Release 0.5.0

### New features

#### Qubit conditioned symbolic operator

- Introduces the QubitConditioned symbolic operator and associated hqml.qcond method. (#23)
  - This performs the symbolic transform `qcond(e^{-itG}) -> e^{-itZG}`, but can generalize to multiple qubits and hooks into the experimental graph decomposition system to enable more advanced circuit decomposition.

### Improvements

- Upgrades to bosonic qiskit v15 (#25)
  - With this new version, you can now simulate `SQR` gates and it should improve handling of `Adjoint(S/T)` gates.

### Other changes

- Adds a pre-commit hook for the Ruff linter/formatter.
- Internally reworks the wire type system to be more flexible.
- Pins the development version of Python to 3.13

### Contributors

This release contains contributions from:
Jim Furches
