# Release 0.6.0

### New features

#### Dynamic qubit allocation

- Adds dynamic qubit allocation as part of the compilation procedure to the `sandiaqscout.hybrid` device (#28)

  This allows for synthesizing qumode gates like `D` in terms of hybrid gates that are native to the device, without having to specify the qubit allocation beforehand. The compiler will automatically allocate qubits as available.

  ```python
  dev = qml.device("sandiaqscout.hybrid", optimize=True, n_qubits=6)
  
  @qml.set_shots(20)
  @qml.qnode(dev)
  def circuit(dist):
      qml.H("q")
      hqml.CD(dist, 0, ["q", "m"])
      hqml.D(dist, math.pi / 2, "m")
      hqml.CD(-dist, 0, ["q", "m"])
      hqml.D(-dist, math.pi / 2, "m")
      qml.H("q")
      return hqml.expval(qml.Z("q"))
  ```

- This uses the new decomposition rule `make_gate_with_ancilla_qubit` that uses the qubit conditioned version of a gate plus dynamic qubit allocation to implement the original gate. This is a general decomposition rule that can be used for any gate, and is not specific to the `sandiaqscout.hybrid` device.
- Adds a decomposition from Pennylane's `FockState` operation to the Hybridlane `FockState` using dynamic qubit allocation (#28)

### New demos

- Two new templates, the `SqueezedCatState` and `GKPState` templates are available, implementing the gadgets from the non-Abelian QSP paper (arxiv:2504.19992) (#28)

  These templates can be used to prepare non-Gaussian states that are useful for quantum error correction and other applications.

  ```python
  fock_level = 256
  dev = qml.device("bosonicqiskit.hybrid", max_fock_level=fock_level)

  @qml.qnode(dev)
  def circuit(codeword, delta=1):
      GKPState(delta, logical_state=codeword, wires=["q", "m"])

      # SBS stabilizer measurement
      alpha = np.sqrt(np.pi / 8)
      lam = -alpha * delta**2
      hqml.YCD(lam, 0, wires=["q", "m"])
      hqml.XCD(2 * alpha, np.pi / 2, wires=["q", "m"])
      hqml.YCD(lam, 0, wires=["q", "m"])
      return hqml.expval(qml.Z("q"))

  stabilizer_expval = circuit(codeword, 0.34)
  ```

  They also use PennyLane's new algorithmic error calculation, allowing one to calculate algorithm error resulting from approximations:

  ```python
  fock_level = 256
  dev = qml.device("bosonicqiskit.hybrid", max_fock_level=fock_level)

  @qml.qnode(dev)
  def circuit(alpha):
      SqueezedCatState(alpha, np.pi / 2, wires=["q", "m"])

      qml.H("a")
      hqml.ConditionalParity(["a", "m"])
      qml.H("a")
      return hqml.expval(qml.Z("a"))  

  errors_dict = qml.resource.algo_error(circuit)(alpha)
  error = errors_dict["SpectralNormError"]
  ```

### Improvements

- Adds many aliases for hybrid and CV gates, such as `hqml.D`, `hqml.CD`, etc (#28)
- Support for bibtex in documentation, updates the readme, and adds the contributing guide (#28)
- Doctest is introduced to ensure documentation is up to date (#28)
- Slow marker added to pytest. Skip slow simulation-based tests with `uv run pytest -m "not slow"` (#28)

### Breaking changes

- Changed the hardware wire notation for the `sandiaqscout.hybrid` device to now use the "manifold" notation. (#28)

  Previously, hardware qumodes were addressed with axis-mode notation like "a0m1", but with the device now using "manifolds", this has been updated to `m<manifold>i<index>` where `manifold` takes the role of `axis` in determining the radial direction of the qumodes, and `index` takes the role of `mode` in determining which mode in that direction to address. Note that there is also now a `Qumode` object that can be used as a wire to more explicitly specify the addressing, like `Qumode(1, 2)`.

  To update prior code, change `axis {0, 1} -> manifold {1, 0}` and keep the index the same, so a wire `a0m1` (tilt mode on axis 0) is now `m1i1` (tilt mode on manifold 1).

- Renames `hqml.FockLadder` to `hqml.FockState` (#28)
- Modifies OpenQASM library `cvstdgates.inc` to add wire types in function definitions (#28)
- OpenQASM exporting no longer uses constants for homodyne/pnr precision; instead, it uses the hardware's native precision (#28)
- Updates to PennyLane 0.44 (#28)

### Fixes

- Fixes operator constraint encoding in the `sandiaqscout.hybrid` device to handle `Adjoint(.)` gates (#28)
- `adjoint()` method of CD and CS gates (#28)
- Decomposition of ECD gate (#28)
- Uses `typing_extensions` for `override` to remain compatible with Python 3.11 (#27)

### Contributors

This release contains contributions from:
Jim Furches, Carlos Ortiz Marrero, Blake Burgstahler
