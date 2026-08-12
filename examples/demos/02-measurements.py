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
    ## Measurements

    Measurements in hybridlane follow those in PennyLane as best as we can, so we recommend reading their [documentation](https://docs.pennylane.ai/en/stable/introduction/measurements.html). We'll focus on highlighting the differences and what's supported.

    The currently supported measurements are:

    - `hl.expval(obs)`: The expectation value $\langle O \rangle$ of an observable
    - `hl.var(obs)`: The variance of an observable $Var[O] = \langle O^2\rangle - \langle O \rangle^2$
    - `hl.sample(obs, map)`: Sample eigenvalues of an observable `obs` _or_ sample computational basis states specificed by `map`
    - `hl.state()`: The full state vector $\ket{\psi}$ at the end of the circuit (simulator-only)
    - `hl.density_matrix(wires)`: The density matrix $\rho$ at the end of the circuit, possibly keeping only the wires in `wires` (simulator-only)

    ## Observables

    Here's a list of the common observables you might be interested in

    - `qp.{X,Y,Z}`: The qubit Pauli matrices $X, Y, Z$
    - `qp.Projector`: A projector $\ket{\phi}\bra{\phi}$ onto an arbitrary qubit state
    - `hl.{N,P,X}`: The CV observables $\hat{n}, \hat{x}, \hat{p}$.
    - `hl.QuadOperator`: The rotated observable $\hat{x}_\phi$
    - `hl.FockStateProjector`: A projector onto a multi-mode Fock basis state $\ket{n_1,n_2,\dots}\bra{n_1,n_2,\dots}$

    As an example of how these can be used, let's show the famous Heisenberg uncertainty relation $\sigma_x \sigma_p \geq \hbar/2$ for a squeezed state:
    """)
    return


@app.cell
def _():
    import numpy as np
    import pennylane as qp

    import hybridlane as hl

    dev = qp.device("default.hybrid", fock_level=64)

    @qp.qnode(dev)
    def squeezed_circuit(r):
        hl.S(r, 0, wires=0)
        return hl.var(hl.X(0)), hl.var(hl.P(0))

    for r in (-1.0, 0, 1.0):
        var_x, var_p = squeezed_circuit(r)
        val = np.sqrt(var_x * var_p)
        print(f"Squeezing param {r}: {val} >= 1/2")
    return dev, hl, np, qp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see, up to numerical error the squeezed state is saturating the bound for each squeezing parameter $r$.

    It's worth emphasizing that the observables above can be manipulated through [symbolic math](https://docs.pennylane.ai/en/stable/news/new_opmath.html). For example, we can produce higher powers of the CV observables like `hl.N(w) ** 2` for $\hat{n}^2$ or take tensor products of qubit and qumode observables like `qp.Z(0) @ hl.P(1)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## State measurements

    In simulators, you can directly return the state of the circuit after all the operations have been applied. `hl.state` always returns the full state vector over all the wires, in the subsystem order determined by the order of appearance of each wire in the circuit. `hl.density_matrix` lets you optionally discard unwanted wires, with full control over the ordering of the remaining subsystems.

    For each, the resulting array size is determined by the truncation of your device. Additionally, while we don't have any such simulators at the moment, in principle the basis of the state vector may be different between different simulators (for example, if one is in Fock space and the other simulates on a position grid).

    Here's an example to illustrate how `hl.state` works on `default.hybrid`, which simulates in Fock space.
    """)
    return


@app.cell
def _(dev, hl, qp):
    @qp.qnode(dev)
    def state_circuit():
        qp.CatState(0.123, 0, 0, wires=1)
        qp.X(0)
        qp.H(2)

        return hl.state()

    psi = state_circuit()
    return psi, state_circuit


@app.cell(hide_code=True)
def _(mo, psi):
    mo.md(r"""
    The dimension of the state vector is $dim(\ket{{\psi}}) = 256$. The subsystem ordering is that determined by the order the wires appear in the `QuantumScript`. If you're unsure, you can determine this with
    """)
    return


@app.cell
def _(qp, state_circuit):
    tape = qp.workflow.construct_tape(state_circuit)()
    tape.wires
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because we created a separable state, we could express it as $\ket{\psi} = \ket{C_\alpha} \otimes \ket{1} \otimes \ket{+}$.

    Next, we'll show how to work with `hl.density_matrix`. Consider the following circuit
    """)
    return


@app.cell
def _(dev, hl, qp):
    @qp.qnode(dev)
    def circuit(x):
        qp.H(0)
        hl.CD(x, 0, wires=(0, 1))
        return hl.density_matrix(1)  # keeps wire 1, the qumode

    rho = circuit(2.0)
    rho.shape
    return (rho,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We obtain the resulting qumode state (using a truncation of 64 levels per qumode), discarding the qubit used to prepare the superposition. We can inspect its eigenvalues to show that it's consistent with $\rho_1 \approx 0.5 \ket{\alpha}\bra{\alpha} + 0.5 \ket{-\alpha}\bra{-\alpha}$
    """)
    return


@app.cell
def _(np, rho):
    e = np.linalg.eigvalsh(rho)
    e[np.abs(e) > 1e-2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The argument to `hl.density_matrix(wires)` can be multiple wires, allowing you to keep multiple qubits/qumodes. The resulting subsystem order of the density matrix will be the same as `wires`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## hl.sample

    `hl.sample` allows you to sample the eigenvalues of an observable or to sample computational basis states. It only works in finite-shot mode, which you enable with `qp.set_shots`. Here's an example performing photon number readout of a coherent state using the observable mode:
    """)
    return


@app.cell
def _(dev, hl, qp):
    @qp.set_shots(10)
    @qp.qnode(dev)
    def coherent_state(a):
        qp.CoherentState(a, 0, wires=0)
        return hl.sample(hl.N(0))

    coherent_state(2.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is sampling eigenvalues $n \sim |\langle n|\psi\rangle|^2$. For a more complex circuit, you may instead wish to read off basis states per-shot, such as performing quantum phase estimation. In hybridlane, the sample function is a little more complicated than PennyLane because we have multiple choices of basis state for qumodes. Here's an example of taking shots over a hybrid system
    """)
    return


@app.cell
def _(dev, hl, qp):
    from pprint import pprint

    from hybridlane.wires import ComputationalBasis

    @qp.set_shots(10)
    @qp.qnode(dev)
    def cat_state_sampled(a):
        qp.H(0)
        hl.CD(a, 0, wires=(0, 1))

        map = hl.wires.BasisMap({(0, 1): ComputationalBasis.Discrete})
        return hl.sample(schema=map)

    result = cat_state_sampled(1.0)
    pprint(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The returned object is much more complex than in PennyLane. The `data` field will be a dict containing an array of shape `(shots,)` per wire. We can't group them together because in principle, sampling different wires may yield different data types, and most tensor libraries don't allow mixed `dtype`. For example, if working on a device that could measure a qubit and perform homodyne measurement on a qumode, then sampling the qubit yields a `bool` or `int` type, and the homodyne measurement yields a `float` type.

    For qubits and qudits, you always use the type `ComputationalBasis.Discrete`, while for qumodes you have two primary options:

    - Photon-number readout: `ComputationalBasis.Discrete`
    - Homodyne measurement: `ComputationalBasis.Position`

    We can't illustrate a sampled homodyne measurement here because it's not supported by `default.hybrid`.
    """)
    return


if __name__ == "__main__":
    app.run()
