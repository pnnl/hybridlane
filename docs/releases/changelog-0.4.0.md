# Release 0.4.0

### New features

#### Sandia QSCOUT ion trap

- Adds the `sandiaqscout.hybrid` device that serves as a compilation target for a trapped ion quantum computer and includes definitions for native gates (#16)
- Introduces the `to_jaqal` function to export to the native IR

### New demos

- Several new demos added for the workshop at NCSU titled _Introducing Hybridlane_ (#17)
- New `FockLadder` template to prepare a definite Fock state with sequences of Blue/Red sideband pulses (#16)

### Graph decomposition

- Implements many gate semantics like `Adjoint` and `Pow` and gate decompositions (from arXiv:2409.03747) in the experimental Pennylane graph decomposition system (#17)

### Improvements

- Improves inference of wire types for `Controlled` gates (#17)

### Breaking changes

- Renames Bosonic Qiskit device to `bosonicqiskit.hybrid` (#17)
- Now only measured wires will have results for `hqml.sample` in Bosonic Qiskit (#17)
- Improves handling of int parameters in `to_openqasm` by printing with float format (#17)
- Suppresses `SparseEfficiencyWarning` when simulating in Bosonic Qiskit (#17)

### Fixes

- Corrects the translation for the `ConditionalRotation` gate to Bosonic Qiskit (#17)
- Removes the global phase from the `ModeSwap` decomposition (#17)

### Other changes

- Refactors code and removes features deprecated in Python 3.10 (#17)

### Contributors

This release contains contributions from:
Jim Furches
