# Release 0.8.0

### New features

#### `default.hybrid` device 🖥️

- A new simulator device called `default.hybrid` that is compatible with NumPy and JAX is now available. `default.hybrid` supports `jax.jit` compilation for faster execution and is differentiable. It should be fully compatible with all existing simulations run on `bosonicqiskit.hybrid` and is compatible with many more classes of circuits such as qubit-only circuits and qutrit-based circuits. (#62) (#65) (#66)

    Consider the logical cat state readout circuit from [Putterman et al., Nature 638](https://www.nature.com/articles/s41586-025-08642-7)
    
    ```python
    import numpy as np
    import pennylane as qp
    import hybridlane as hl
    
    dev = qp.device("default.hybrid", fock_level=32)

    @qp.qnode(dev)
    def circuit(alpha):
        qp.CatState(alpha, 0, 0, wires=0)
        hl.D(alpha, 0, 0)
        qp.H(1)
        hl.SQR(np.pi, np.pi / 2, 0, wires=[1, 0])
        qp.H(1)
        
        return hl.expval(qp.Z(1))
    ```

    ```python
    >>> circuit(2.0)
    np.float64(-0.0003354626279027384)
    ```
    
    The device is compatible with `@jax.jit` and automatic differentiation. Let's define an ansatz of SNAP and Displacement gates, followed by a loss function that calculates the infidelity between our prepared state and the logical binomial code state |0L>
    
    ```python
    import pennylane as qp
    import hybridlane as hl
    
    import jax
    import optax

    jax.config.update("jax_enable_x64", True)
    
    fock_level = 20
    dev = qp.device("default.hybrid", fock_level=fock_level)
    
    @qp.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(params):
        @qp.for_loop(0, 5)
        def loop_body(i):
            hl.D(params[i, 0], params[i, 1], wires=0)
    
            for j in range(8):
                hl.SNAP(params[i, 2 + j], j, wires=0)
    
        loop_body()
        return hl.state()
    
    # Target state is the binomial codeword |0_L> = (|0> + |4>)/sqrt(2)
    codeword = hl.math.concatenate(
        [
            hl.math.array([1, 0, 0, 0, 1], like="jax") / np.sqrt(2),
            hl.math.zeros(fock_level - 5, like="jax"),
        ]
    )
    
    @jax.jit
    def loss(params):
        state = circuit(params)
        return 1 - hl.math.real(hl.math.fidelity_statevector(state, codeword))
    ```

    Now we can JIT the optimization end-to-end using Optax

    ```python
    key = jax.random.key(0)
    params = jax.random.normal(key, (5, 10))
    starting_loss = loss(params)
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state):
        loss_value, grads = jax.value_and_grad(loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    for _ in range(100):
        params, opt_state = train_step(params, opt_state)
    ```

    ```python
    >>> starting_loss
    Array(0.7311361, dtype=float64)
    >>> loss(params)
    Array(0.00493101, dtype=float64)
    >>> circuit(params).round(4)
    Array([-6.342e-01-0.3301j,  4.500e-03-0.0018j, -2.700e-02+0.009j ,
           -0.000e+00-0.0038j, -6.169e-01-0.3217j,  4.800e-03-0.0015j,
           -8.500e-03-0.0043j,  3.810e-02-0.0026j, -1.400e-03-0.0238j,
           -1.100e-02-0.0016j,  1.710e-02+0.0173j, -1.630e-02-0.0193j,
            1.050e-02+0.0137j, -3.900e-03-0.0065j, -6.000e-04+0.0009j,
            2.400e-03+0.0021j, -2.400e-03-0.0029j,  1.700e-03+0.0027j,
           -9.000e-04-0.0017j,  5.000e-04+0.0015j], dtype=complex128)
    ```

    The final infidelity of `0.00493` is much lower than the initial starting infidelity of `0.731`.

    The device also has experimental support for simulating qutrits from PennyLane's `qp.ops.qutrit` module. Using the `QutritBasisState`, we prepare the state `|1>|2>` on 2 qutrits. Then we swap them with `TSWAP` to obtain `|2>|1>`

    ```python
    import pennylane as qp
    import hybridlane as hl

    dev = qp.device("default.hybrid", fock_level=4)

    @qp.qnode(dev)
    def circuit():
        qp.QutritBasisState([1, 2], wires=[0, 1])
        qp.TSWAP(wires=[0, 1])
        return hl.state()
    ```

    ```python
    >>> circuit()
    array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j,
           0.+0.j])
    ```

    Indeed the expected nonzero index of the state is `2 * 3 + 1 = 7`.

#### Fock and symplectic representations

- Each operator now has NumPy and JAX-compatible definitions for obtaining the matrix representation of the operator in the Fock basis (#41) (#76)

    This function works for all operators inheriting from `FockRepresentation`. The `fock_matrix` function accepts a dictionary mapping each wire to its dimension

    ```python
    >>> hl.R(0.5, wires=0).fock_matrix({0: 3})
    array([[1.        +0.j        , 0.        +0.j        ,
            0.        +0.j        ],
           [0.        +0.j        , 0.87758256-0.47942554j,
            0.        +0.j        ],
           [0.        +0.j        , 0.        +0.j        ,
            0.54030231-0.84147098j]])
    ```

  See also the `hl.fock_matrix` function, which does the analogous operation to `qp.matrix`, and can construct the matrix for entire circuits.

- The symplectic representations (`_heisenberg_rep`) of the Gaussian operations have also been updated (#71)

    The matrix uses the same basis as PennyLane `(1, x1, p1, x2, p2, ...)` in "xpxp" ordering:

    ```python
    >>> hl.S(0.5, 0, wires=0).heisenberg_tr((0,))
    array([[1.        , 0.        , 0.        ],
           [0.        , 0.60653066, 0.        ],
           [0.        , 0.        , 1.64872127]])
    ```

    The new module `hl.math.symplectic` contains utilities for converting between the phase-space symplectic basis `(1, x1, p1, ...)` and the equivalent mode basis `(1, a1, ad1, a2, ad2, ...)` using the `to_fock_space` and `to_phase_space` functions.

    ```python
    >>> S = hl.D(0.5, 0, wires=0).heisenberg_tr((0,))
    >>> S
    array([[1.        , 0.        , 0.        ],
           [0.70710678, 1.        , 0.        ],
           [0.        , 0.        , 1.        ]])
    >>> hl.math.to_fock_space(S)
    array([[1. +0.j, 0. +0.j, 0. +0.j],
           [0.5+0.j, 1. +0.j, 0. +0.j],
           [0.5+0.j, 0. +0.j, 1. +0.j]])
    ```

### Improvements

- New `hl.math` module adapts familiar PennyLane `qp.math` functions to heterogeneous system dimensions (#40) (#43) (#45)

- Density matrix extraction from simulation is now supported with `hl.density_matrix` (#62)

    ```python
    import pennylane as qp
    import hybridlane as hl

    fock_levels = 8
    dev = qp.device("default.hybrid", fock_level=fock_levels)

    @qp.qnode(dev, interface=like)
    def circuit():
        qp.H(0)
        hl.CD(2.0, 0, wires=[0, 1])
        return hl.density_matrix(wires=1)
    ```

    ```python
    >>> rho = circuit()
    >>> rho.shape
    (8, 8)
    >>> hl.math.diag(rho)
    array([0.01830246+0.j, 0.0734484 +0.j, 0.14531038+0.j, 0.20032632+0.j,
           0.1817865 +0.j, 0.18621053+0.j, 0.0636499 +0.j, 0.1309655 +0.j])
    ```

    As expected, we get a `8x8` density matrix matching the qumode's dimension (`fock_level`), and we get a decent spread over the probability of measuring each basis state.

- Rebranded hybridlane's import to `hl`, matching PennyLane's switch to `qp` (#45)

- Cleaned up the sidebar in the GitHub pages documentation (#63)

- Support for detecting qutrits from PennyLane's `qp.ops.qutrit` module in the type system (#66)

- Simulator and hardware device versions are now tied to hybridlane's version (#61)

- Updated hybridlane to depend on PennyLane v0.45 (#47)

### Breaking changes

- Refactored the type checker code from `hl.sa` to `hl.wires` module and renamed a few functions/classes (#66)

- Updated the convention for the `Squeezing` and `ConditionalSqueezing` gates to match [Liu et al., PRX Quantum 7, 101201](https://journals.aps.org/prxquantum/abstract/10.1103/4rf7-9tfx) (#69)

    The squeezing parameter `r` is unaffected, but the angle parameter has now been updated from `zeta = r e^{i\theta} -> zeta = r e^{i2\theta}`.

### Fixes

- Fixed the `fock_spectrum` function for `hl.FockStateProjector` (#55)

- `wire_icon_colors` keyword argument is now properly passed down when calling `hl.draw_mpl` on a non-QNode callable (#72)

- The `simplify()` method of the JC and AJC gates is no longer periodic in theta (#74)

### Miscellaneous

- Added a `justfile` to facilitate running common developer tasks (#42)

- New test markers to facilitate granular test execution (#39) (#42)

- Dependabot is now enabled to help keep code up to date (#56) (#57) (#58) (#59) (#60) (#70)

- Added `jax`, `optax`, and `ty` to developer dependencies (#47) (#62)

- Drops dependency on `pydantic` (#84)

- Improved CI testing and linting (#86) (#87)

### Contributors

This release contains contributions from:
Jim Furches
