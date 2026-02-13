# Release 0.3.0

### New features

#### Circuit drawing

- Added circuit drawing function `hqml.draw_mpl` (#15)
  - Circuit diagrams are now annotated by default with qubit and qumode icons, and they display qubit-conditioned gates with a special notation. See the docs for more details.

#### Improvements
- Truncation arguments were simplified for `hybrid.bosonicqiskit` device (#14)
  - Qumodes and wires are no longer needed, instead the device infers the qubits and qumodes from the circuit structure and can automatically apply `max_fock_level` to each qumode.

### Contributors

This release contains contributions from:
Jim Furches
