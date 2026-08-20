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
    # CV-DV state transfer

    hybridlane provides two templates that move a quantum state between a qumode and a register of qubits, implementing quantum analog-to-digital (`StateTransferCVtoDV`) and digital-to-analog (`StateTransferDVtoCV`) conversion via non-Abelian QSP.

    Both act on the wires `(q0, q1, ..., q_{n-1}, qumode)`, cost exactly $2n$ conditional displacements, and share a $2^n$-point position grid

    $$q_s = \lambda\left(2s - (2^n - 1)\right), \qquad s = 0, \dots, 2^n - 1,$$

    with spacing $2\lambda$. The spacing parameter $\Delta$ of the reference below is $2\lambda$.

    Y. Liu, J. M. Martyn, J. Sinanan-Singh, K. C. Smith, S. M. Girvin and I. L. Chuang, "Toward Mixed Analog-Digital Quantum Signal Processing: Quantum AD/DA Conversion and the Fourier Transform", *IEEE Transactions on Signal Processing*, vol. 73, 2025, [doi:10.1109/TSP.2025.3599462](https://doi.org/10.1109/TSP.2025.3599462), Sec. IV-B.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pennylane as qp

    import hybridlane as hl

    qp.decomposition.enable_graph()
    return hl, np, qp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CV $\to$ DV

    With the qubits in $\ket{0\cdots0}$ and the qumode holding $\ket{\psi}$, measuring the qubit register samples the position wavefunction, $P(s) \approx |\psi(q_s)|^2 \cdot 2\lambda$. Reading out the vacuum should reproduce a Gaussian on the grid:
    """)
    return


@app.cell
def _(hl, np, qp):
    from hybridlane.wires import BasisMap, ComputationalBasis

    n_ad, lmbda, shots = 4, 0.29, 4096
    qubits_ad, mode_ad = list(range(n_ad)), n_ad

    dev_ad = qp.device("bosonicqiskit.hybrid", max_fock_level=64)

    @qp.set_shots(shots)
    @qp.qnode(dev_ad)
    def read_out_vacuum():
        hl.Displacement(0.0, 0.0, mode_ad)  # qumode stays in vacuum
        hl.StateTransferCVtoDV(n_ad, lmbda, wires=[*qubits_ad, mode_ad])
        schema = BasisMap(
            {qp.wires.Wires(w): ComputationalBasis.Discrete for w in qubits_ad}
        )
        return hl.sample(schema=schema)

    result = read_out_vacuum()
    counts = np.zeros(2**n_ad)
    for i in range(shots):
        counts[int("".join(str(result.data[w][i]) for w in qubits_ad), 2)] += 1
    measured = counts / shots

    grid = lmbda * (2 * np.arange(2**n_ad) - (2**n_ad - 1))
    ideal = np.exp(-(grid**2))
    ideal = ideal / ideal.sum()

    f"classical fidelity against |psi(q_s)|^2: {np.sum(np.sqrt(measured * ideal)) ** 2:.4f}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DV $\to$ CV

    Run backwards, the protocol moves the state off the qubits and into the qumode. It is coherent and deterministic, but requires the qumode to *start* in the sinc state $\ket{0, \Delta}^{\rm sinc} \propto \int dq \, \mathrm{sinc}(\pi q / \Delta) \ket{q}$. That state has infinite energy, so a squeezed vacuum stands in for it.

    Two numbers report success, both approaching 1: the population of $\ket{0\cdots0}$ left on the register, and the purity of the qumode.
    """)
    return


@app.cell
def _(hl, mo, np, qp):
    n_da, delta, squeezing, fock_da = 3, 1.2, 2.5, 256
    qubits_da, mode_da = list(range(n_da)), n_da

    dev_da = qp.device("bosonicqiskit.hybrid", max_fock_level=fock_da)

    @qp.qnode(dev_da)
    def transfer_to_qumode(amps, r):
        qp.StatePrep(amps, wires=qubits_da)
        if r:
            hl.Squeezing(r, 0.0, mode_da)
        hl.StateTransferDVtoCV(n_da, delta / 2, wires=[*qubits_da, mode_da])
        return hl.state()

    def diagnose(amps, r):
        amps = np.asarray(amps, dtype=complex)
        vec = np.asarray(
            transfer_to_qumode(amps / np.linalg.norm(amps), r)
        ).reshape((2,) * n_da + (fock_da,))

        zeros = vec[(0,) * n_da]
        flat = vec.reshape(-1, fock_da)
        rho = flat.conj().T @ flat
        rho = rho / np.trace(rho)
        return (
            float(np.real(np.vdot(zeros, zeros))),
            float(np.real(np.trace(rho @ rho))),
        )

    plus = np.ones(2**n_da)
    pop_sq, pur_sq = diagnose(plus, squeezing)
    pop_vac, pur_vac = diagnose(plus, 0.0)

    mo.md(f"""
    Transferring $\\ket{{{'+' * n_da}}}$ with $n = {n_da}$, $\\Delta = {delta}$:

    | input qumode | $P(\\ket{{0\\cdots0}})$ | purity |
    |---|---|---|
    | squeezed, $r = {squeezing}$ | {pop_sq:.4f} | {pur_sq:.4f} |
    | vacuum | {pop_vac:.4f} | {pur_vac:.4f} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Starting from the vacuum leaves a third of the population off $\ket{0\cdots0}$ and the qumode entangled with the register. Squeezing too far is no better, since the state stops fitting under the Fock cutoff.
    """)
    return


if __name__ == "__main__":
    app.run()
