import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## State preparation

    A common workflow in experiments is to approximate some desired quantum state $\ket{\phi}$ by fixing a circuit structure that's amenable to hardware and optimizing its parameters numerically. In this notebook, we'll demonstrate the tools hybridlane has to enable this workflow.

    Our goal for this first example will be to prepare the binomial code state $\ket{\phi} = \frac{1}{\sqrt{2}}(\ket{0} + \ket{4})$ on a qumode, and we'll use the universal gate set of $\{SNAP, D\}$. Let's define the ansatz as interleaving $SNAP$ (up to 8 Fock levels) and $D$ gates on the qumode:
    """)
    return


@app.cell
def _():
    import pennylane as qp
    from pennylane.typing import TensorLike

    import hybridlane as hl

    # snap_ansatz takes an array of shape (layers, 10) as input
    def snap_ansatz(x: TensorLike):
        layers = hl.math.shape(x)[0]

        @qp.for_loop(0, layers)
        def loop_body(i):
            hl.D(x[i, 0], x[i, 1], wires=0)

            # hybridlane's SNAP gate acts on a single Fock level `j`, so we chain 8 of them
            # to manipulate the lowest 8 energy levels of the qumode.
            for j in range(8):
                hl.SNAP(x[i, 2 + j], j, wires=0)

        loop_body()

    return hl, qp, snap_ansatz


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's define our optimization objective, which will be to maximize the fidelity $F(\ket{\phi}, \ket{\psi(\theta)})$.
    """)
    return


@app.cell
def _(hl, qp, snap_ansatz):
    fock_level = 20
    dev = qp.device("default.hybrid", fock_level=fock_level)

    @qp.qnode(dev)
    def state_prep(x):
        snap_ansatz(x)
        return hl.state()

    def loss(x, phi):
        psi = state_prep(x)
        return 1 - hl.math.fidelity_statevector(phi, psi)

    return dev, fock_level, loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With that in place, we can optimize our circuit using Scipy. Here we're using a small number of iterations so that the notebook runs quickly.
    """)
    return


@app.cell
def _(fock_level, hl, loss):
    import numpy as np
    from scipy.optimize import minimize

    def _():
        # Create random parameters for 3 layers
        rng = np.random.default_rng(42)
        x = rng.standard_normal((3, 10))

        # Instantiate the binomial state
        phi = hl.math.concatenate(
            [
                hl.math.array([1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2)], like=x),
                hl.math.zeros(fock_level - 5, like=x),
            ]
        )

        # Scipy requires a 1D array
        def scipy_loss(x):
            return loss(x.reshape(3, 10), phi)

        result = minimize(scipy_loss, x.flat, method="L-BFGS-B", options=dict(maxiter=10))
        return result

    scipy_result = _()
    scipy_result
    return np, scipy_result


@app.cell(hide_code=True)
def _(mo, scipy_result):
    mo.md(rf"""
    With our 10 BFGS steps, we were able to reach an infidelity of $0.103$, and the optimized circuit parameters could be obtained from `x`.

    ## Speeding up with `jax.jit`

    The above example is very slow because on each function invocation (above {scipy_result.nfev} of them to be precise), the circuit structure must be reconstructed and simulated. However the structure isn't changing, and it'd be nice to reuse it across iterations. To speed up optimization workflows like this, hybridlane's `default.hybrid` device supports JAX, which will trace the computation once and produce a highly optimized native binary that _does_ reuse the circuit structure.

    We'll now modify the above code to be JIT-compatible. The first thing is to make sure that `default.hybrid` is using `diff_method = "backprop"` to let JAX take gradients through the entire simulation.
    """)
    return


@app.cell
def _(dev, hl, qp, snap_ansatz):
    @qp.qnode(dev, interface="jax", diff_method="backprop")
    def state_prep_jax(x):
        snap_ansatz(x)
        return hl.state()

    def loss_jax(x, phi):
        psi = state_prep_jax(x)
        return 1 - hl.math.fidelity_statevector(phi, psi).real

    return (loss_jax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we have to write a little more code than Scipy to produce the optimizer. To be compatible with JAX, we need to write it in a functional style.
    """)
    return


@app.cell
def _(fock_level, hl, loss_jax, np):
    import jax
    import jax.numpy as jnp
    import optax

    # By default, jax does everything in f32, but we want double precision
    jax.config.update("jax_enable_x64", True)

    # With `jax.jit`, this entire comptation will be traced. Because `opt` and `maxiter` are not
    # tensors, we must declare them to be "static"
    @jax.jit(static_argnames=("opt", "loss", "maxiter"))
    def optimize_jax(opt, x0, loss, maxiter):
        # We explicitly tell jax how to update at each step. Following the
        # semantics of `jax.lax.fori_loop`, this inner function takes two parameters,
        # the loop iteration `i` (unused), and `val`, the state to be carried through
        # each iteration.
        def update(i, val):
            x, opt_state = val
            _, grads = jax.value_and_grad(loss)(x)
            updates, opt_state = opt.update(grads, opt_state)
            x = optax.apply_updates(x, updates)
            return x, opt_state

        val = (x0, opt.init(x0))
        x, opt_state = jax.lax.fori_loop(0, maxiter, update, val)
        return x

    def _():
        # Create parameters with same shape as before
        key = jax.random.key(42)
        x0 = jax.random.normal(key, (3, 10))

        # Instantiate our optimizer, we'll choose the adam optimizer
        opt = optax.adam(learning_rate=1e-2)

        # Instantiate the binomial state
        phi = hl.math.concatenate(
            [
                hl.math.array([1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2)], like=x0),
                hl.math.zeros(fock_level - 5, like=x0),
            ]
        )

        # Perform the optimization
        starting_loss = loss_jax(x0, phi)
        x = optimize_jax(opt, x0, loss=lambda x: loss_jax(x, phi), maxiter=1000)
        final_loss = loss_jax(x, phi)

        return starting_loss, final_loss

    starting_loss, final_loss = _()
    print(f"Starting loss: {starting_loss}")
    print(f"Final loss: {final_loss}")
    return jax, optax, optimize_jax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using JAX, we were able to achieve a much smaller loss in just a fraction of the time it took Scipy.

    ## Unitary synthesis

    The above ideas extend to performing numerical unitary synthesis too: define your circuit architecture with hybridlane and then throw it into a numerical optimizer. To extract the unitary matrix implemented by your circuit, hybridlane provides the [hl.fock_matrix](https://pnnl.github.io/hybridlane/_autoapi/hybridlane/ops/functions/fock_matrix/index.html#module-hybridlane.ops.functions.fock_matrix) function.

    To demonstrate this, we'll attempt to produce the nonlinear Kerr gate $K(\theta) = e^{-i\theta\hat{n}^2}$ using linear $CR(\theta) = e^{-i\theta Z\hat{n}/2}$ gates and single-qubit rotations.
    """)
    return


@app.cell
def _(fock_level, hl, qp):
    wire_dims = {0: 2, 1: fock_level}

    # x has shape (layers, 5)
    def cr_ansatz(x):
        layers = hl.math.shape(x)[0]

        @qp.for_loop(0, layers)
        def loop_body(i):
            qp.Rot(*x[i, 0:3], wires=0)
            hl.CR(x[i, 3], wires=(0, 1))

        loop_body()

    def unitary(x):
        return hl.fock_matrix(cr_ansatz, wire_order=(0, 1), wire_dims=wire_dims)(x)

    return unitary, wire_dims


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now define the loss function based on the unitary fidelity

    $$F(U, V) = \frac{1}{d^2} \left|\text{Tr}[U^\dagger V]\right|^2$$

    where $d$ is the dimension of our Hilbert space.
    """)
    return


@app.cell
def _(hl, unitary):
    def fidelity(U, V):
        d = hl.math.shape(U)[0]
        norm = hl.math.trace(hl.math.dag(U) @ V)
        return (hl.math.abs(norm) ** 2).real / d**2

    def cr_loss(x, V):
        U = unitary(x)
        return 1 - fidelity(U, V)

    return cr_loss, fidelity


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And finally, we can use a similar optimizer loop as before
    """)
    return


@app.cell
def _(cr_loss, fidelity, hl, jax, optax, optimize_jax, unitary, wire_dims):
    def _():
        # Create parameters with required shape
        key = jax.random.key(42)
        x0 = jax.random.normal(key, (10, 4))

        # Instantiate our optimizer, we'll choose the adam optimizer
        opt = optax.adam(learning_rate=1e-2)

        # Build our target unitary
        op = hl.K(0.5, wires=1)
        V = op.fock_matrix(wire_dims, wire_order=(0, 1))

        # Perform the optimization
        starting_loss = cr_loss(x0, V)
        x = optimize_jax(
            opt,
            x0,
            loss=lambda x: cr_loss(x, V),
            maxiter=1000,
        )
        final_loss = cr_loss(x, V)

        U = unitary(x)
        f_u = fidelity(U, U)

        return f_u, starting_loss, final_loss

    f_u, starting_loss_u, final_loss_u = _()
    print(f"Starting loss: {starting_loss_u}")
    print(f"Final loss: {final_loss_u}")
    print(f"F(U, U): {f_u}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While we haven't gotten a "good" loss, this does illustrate the workflow to numerically synthesize unitary gates. You might play with the circuit architecture, loss function, and optimizer settings to achieve a better result.
    """)
    return


if __name__ == "__main__":
    app.run()
