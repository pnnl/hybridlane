# Release 0.7.0

### New features

- Support for some state preparation routines in `bosonicqiskit.hybrid` device (#38)

    An example using `qml.CoherentState`:
    ```python
    alpha = 1.5
    dev = qml.device("bosonicqiskit.hybrid", max_fock_level=16)
    
    @qml.qnode(dev)
    def circuit(alpha):
        qml.CoherentState(alpha, 0, wires=0)
        return hqml.expval(hqml.N(0))

    >>> circuit(alpha)
    array(2.24999997+0.j)
    ```

- Support for obtaining `bosonicqiskit.hybrid` state with `hqml.state` (#31)

    ```python
    alpha = 1.5
    dev = qml.device("bosonicqiskit.hybrid", max_fock_level=16)
    
    @qml.qnode(dev)
    def circuit(alpha):
        qml.CoherentState(alpha, 0, wires=0)
        return hqml.state()

    >>> circuit(alpha)
    array([3.24652468e-01+0.j, 4.86978702e-01+0.j, 5.16518913e-01+0.j,
           4.47318500e-01+0.j, 3.35488875e-01+0.j, 2.25052779e-01+0.j,
           1.37816119e-01+0.j, 7.81343950e-02+0.j, 4.14370204e-02+0.j,
           2.07185102e-02+0.j, 9.82765229e-03+0.j, 4.44472299e-03+0.j,
           1.92462151e-03+0.j, 8.00690947e-04+0.j, 3.20990485e-04+0.j,
           1.24319080e-04+0.j])
    ```

### New demos

- Includes the demo notebook developed for APSLOS 2026 (#37)

### Breaking changes

- Makes the SNAP gate bosonic, removing the qubit (#35)
    - The decomposition in terms of SQR gates now uses dynamic qubit allocation, requiring `num_work_wires>=1` in the `decompose` transform.

### Improvements

- Standard units convention for `bosonicqiskit.hybrid` device (#38)
    - For consistency with the ISA paper, this switches the convention to $\hbar = 1$ and sets the device to standard units by default. If you wish to use Wigner units, use the flag `units="wigner"`, e.g.

        ```python
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8, units="wigner")
        ```

- Improvements to QSCOUT Jaqal IR generation and gate optimization (#33, #34)
    - Removes the DV gate import `from qscout.std.v1 usepulses *`
    - Removes the shot count from each `subcircuit` block
    - Improves gate cost estimates for the native displacement gates `xCD` and `yCD` and the macro `zCD`. The decomposition system now weights `xCD` and `yCD` as native gates with cost 1 and `zCD` as cost 3.
    - Updates op codes for JC/AJC gates to use new Jaqal syntax

### Bug fixes

- The `qml.Identity` gate no longer constrains wire types (#34)

### Misc

- Improves testing infrastructure with new Pytest markers (#39, #42)
- Adds some `justfile` recipes for common tasks (#42)

### Contributors

This release contains contributions from:
Jim Furches, Blake Burgstahler
