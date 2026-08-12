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
    ## Symbolic op math

    In this notebook, we'll demonstrate how to perform some symbolic gate decomposition in hybridlane. We heavily leverage PennyLane's graph decomposition system, so you'll need to enable that
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pennylane as qp

    import hybridlane as hl

    qp.decomposition.enable_graph()
    return hl, plt, qp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here are the primary symbolic functions that you can use:

    - [qp.adjoint](https://docs.pennylane.ai/en/stable/code/api/pennylane.adjoint.html): Takes the adjoint of an operator $U \mapsto U^\dagger$
    - [qp.pow](https://docs.pennylane.ai/en/stable/code/api/pennylane.pow.html): Takes powers of an operator, $U \mapsto U^z$
    - [qp.ctrl](https://docs.pennylane.ai/en/stable/code/api/pennylane.ctrl.html): Controls a unitary on the state of one or more qubits, e.g. $U \mapsto \Pi_0 I + \Pi_1 U$
    - [hl.qcond](https://pnnl.github.io/hybridlane/_autoapi/hybridlane/index.html#hybridlane.qcond): Conditions a unitary on the state of a qubit. $U \mapsto \Pi_0 U + \Pi_1 U^\dagger$

    Note that while `hl.qcond` realizes a similar thing to `qp.ctrl`, it's such a common pattern in CV-DV computing that hybridlane adds it as a distinct symbolic operation.

    ## `qp.adjoint`

    PennyLane's `qp.adjoint` function allows taking the inverse of a single gate or an entire quantum function, and hybridlane's gates are programmed to interoperate with it. The canonical way to use it is with its functional form. For an operator `op` taking arguments `*args`, the adjoint can be performed like `qp.adjoint(op) -> f(*args)` -- that is, `qp.adjoint` takes an operator or quantum function and returns a _new_ function accepting the arguments of the original operator or quantum function.

    Here's an example of a single $CR$ gate being inverted:
    """)
    return


@app.cell
def _(hl, qp):
    inv_cr = qp.adjoint(hl.CR)
    inv_cr(0.123, wires=(0, 1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Often, this will be inlined like
    """)
    return


@app.cell
def _(hl, qp):
    qp.adjoint(hl.CR)(0.123, wires=(0, 1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `qp.adjoint` function is _lazy_ by default, so it simply wraps our operator in the `Adjoint` type. This is usually a good thing as it works best with PennyLane's graph decomposition system, but if you'd like it to perform the operation eagerly, then we can pass `lazy=False`
    """)
    return


@app.cell
def _(hl, qp):
    qp.adjoint(hl.CR, lazy=False)(0.123, wires=(0, 1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see, it turned $CR(\theta) \mapsto CR(-\theta)$. This saves you from having to program the adjoint yourself in your circuits. The argument to `qp.adjoint` can also be a quantum function calling several operations, so here's an example of a circuit realizing the identity gate
    """)
    return


@app.cell
def _(hl, plt, qp):
    dev = qp.device("default.hybrid", fock_level=16)

    def custom_op(a):
        qp.H(0)
        hl.CD(a, 0, wires=(0, 1))

    @qp.qnode(dev)
    def circuit(a):
        custom_op(a)
        qp.adjoint(custom_op)(a)

        return hl.expval(hl.N(1))

    def _():
        fig = plt.figure()
        hl.draw_mpl(circuit, level="device", style="sketch", fig=fig)(2.0)
        return fig

    _()
    return (dev,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `qp.adjoint` wrapper took care of reversing the operations and negating each, saving us some work.

    ## `qp.pow`

    `qp.pow(op)` raises an operator to a power. Here's an example:
    """)
    return


@app.cell
def _(hl, qp):
    qp.pow(hl.JC(0.5, 0.123, wires=(0, 1)), z=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It can also be accessed with `**`:
    """)
    return


@app.cell
def _(hl):
    hl.JC(0.5, 0.123, wires=(0, 1)) ** 2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And finally, both of those are lazy, so if want to eagerly evaluate it, then pass `lazy=False`
    """)
    return


@app.cell
def _(hl, qp):
    qp.pow(hl.JC(0.5, 0.123, wires=(0, 1)), z=2, lazy=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `qp.ctrl`

    This has similar usage to `qp.adjoint`. Here's an example of realizing a displacement that's not symmetric about the origin, $D_c(\alpha) = \Pi_0 I + \Pi_1 D(\alpha)$
    """)
    return


@app.cell
def _(dev, hl, plt, qp):
    @qp.qnode(dev)
    def circuit2(a):
        qp.H(1)
        qp.ctrl(hl.D, control=[1])(a, 0, wires=0)
        return hl.expval(hl.X(0))

    def _():
        fig = plt.figure()
        hl.draw_mpl(circuit2, level="device", style="sketch", fig=fig)(2.0)
        return fig

    _()
    return (circuit2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we evaluate it, we'll see we get a nonzero value of $\langle \hat{x} \rangle$, whereas we would obtain 0 with a symmetric conditional displacement $CD$.
    """)
    return


@app.cell
def _(circuit2):
    circuit2(2.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    hybridlane has some symbolic rules that enable it to decompose that gate in terms of the more native $CD$ gate
    """)
    return


@app.cell
def _(circuit2, hl, qp):
    decomposed = qp.decompose(circuit2, gate_set={qp.H, hl.D, hl.CD})
    decomposed_tape = qp.workflow.construct_tape(decomposed)(2.0)
    decomposed_tape.operations
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `hl.qcond`

    This is a special symbolic operation unique to hybridlane. If you express your unitary as $U = e^{-i \theta G}$, then this symbolic operation performs $U \mapsto e^{-i \theta Z \otimes G}$, where $Z$ is the Pauli $Z$ on the conditioning qubit. This symbolic identity comes up often in hybrid CV-DV computing: `qcond(D) -> CD`, `qcond(BS) -> CBS`, and so on. For example,
    """)
    return


@app.cell
def _(hl):
    hl.qcond(hl.D(0.123, 0, wires=0), control_wires=[1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also use it to condition a gate on multiple qubits, like $U = e^{-i \theta Z \otimes Z \otimes G}$, and hybridlane can decompose that using CNOT gates if required
    """)
    return


@app.cell
def _(dev, hl, plt, qp):
    @qp.decompose(gate_set={qp.CNOT, hl.CBS})
    @qp.qnode(dev)
    def qcond_circuit():
        hl.qcond(hl.BS, control_wires=[2, 3])(0.123, 0, wires=(0, 1))
        return hl.state()

    def _():
        fig = plt.figure()
        hl.draw_mpl(qcond_circuit, style="sketch", fig=fig)()
        return fig

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Gate decompositions

    Symbolic op math is really important in the decomposition system. Now we'll talk about how to invoke the decomposition system. In addition to some of the decompositions you've seen above, hybridlane has many more from [Liu et al](https://journals.aps.org/prxquantum/abstract/10.1103/4rf7-9tfx) and [Crane et al](https://arxiv.org/abs/2409.03747).

    You can invoke the decomposition system using the PennyLane transform [qp.decompose](https://docs.pennylane.ai/en/stable/code/api/pennylane.decompose.html), which to first order, accepts your desired gate set. Here's one decomposition for the $CBS$ gate from Crane et al, targetted to superconducting hardware:
    """)
    return


@app.cell
def _(dev, hl, plt, qp):
    @qp.decompose(gate_set={hl.BS, hl.CR})
    @qp.qnode(dev)
    def cbs_circuit():
        hl.CBS(2.0, 0, wires=(0, 1, 2))
        return hl.state()

    def _():
        fig = plt.figure()
        hl.draw_mpl(cbs_circuit, style="sketch", fig=fig)()
        return fig

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can list all available decompositions for a gate using `qp.list_decomps`
    """)
    return


@app.cell
def _(hl, qp):
    qp.list_decomps(hl.CD)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Sometimes, depending on your target gate set, multiple rules can apply. `qp.decompose` lets you weight different target gates and it tries to pick the decomposition with the lowest total cost.

    Gates can also be decomposed using _dynamic qubit allocation_ (sorry, we haven't yet implemented qumode allocation). As an example. some platforms like trapped-ion systems don't natively support the $D$ gate, but they do support a variant of the $CD$ gate, the $XCD$ gate, which displaces w.r.t. a different qubit axis. Thus, if we could find a clean, unused qubit in the state $\ket{0}$, we could then realize a $D$ gate with an $XCD$ gate and some single-qubit gates.

    You can instruct the decomposition system to use dynamic qubit allocation with the `num_work_wires` argument.
    """)
    return


@app.cell
def _(dev, hl, qp):
    from hybridlane.ops.op_math.decompositions.qubit_conditioned_decompositions import (
        make_gate_with_ancilla_qubit,
    )

    # This also shows how to temporarily use some additional decomposition rules. here,
    # make_gate_with_ancilla_qubit is a symbolic rule we made to use hl.qcond() under the hood.
    @qp.decompose(
        gate_set={hl.XCD, qp.H},
        fixed_decomps={hl.D: make_gate_with_ancilla_qubit(hl.D)},
        num_work_wires=None,  # `None` allows as many qubits allocated as necessary
    )
    @qp.qnode(dev)
    def dynamic_d_circuit():
        hl.D(2.0, 0, wires=0)
        return hl.state()

    def _():
        tape = qp.workflow.construct_tape(dynamic_d_circuit)()
        return tape.operations

    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can find more information on how to work with their graph decomposition system [here](https://docs.pennylane.ai/en/stable/introduction/compiling_circuits.html).

    ## Quantum phase estimation

    Putting all this together, let's see how these decomposition identities can be used to synthesize a high-level algorithmic primitive: the quantum phase estimation. We can use it to perform a Fock state readout using qubits (albeit an expensive one). The hamiltonian of our system will be $H = \hat{n}$, whose time evolution is the familiar phase-space rotation $R(\theta) = e^{-i\theta\hat{n}}$.
    """)
    return


@app.cell
def _(dev, hl, qp):
    from pprint import pprint

    from hybridlane.wires import BasisMap, ComputationalBasis

    # We have to build the unitary outside the circuit so it isn't queued
    # into the circuit's operations
    op = hl.R(1.0, wires="m")

    @qp.decompose(
        gate_set={hl.R, hl.CR, qp.CNOT, qp.H, qp.ControlledPhaseShift, qp.SWAP, qp.FockState}
    )
    @qp.set_shots(10)
    @qp.qnode(dev)
    def qpe(n, n_bits):
        # This is just for illustration, you could prepare any state prior
        qp.FockState(n, wires="m")

        wires = range(n_bits)
        qp.QuantumPhaseEstimation(op, estimation_wires=wires)

        # Sample computational bitstrings
        map = BasisMap({wires: ComputationalBasis.Discrete})
        return hl.sample(schema=map)

    bitstrings = qpe(4, n_bits=4)
    pprint(bitstrings)
    return (qpe,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Behind the scenes, many identities have been chained together to realize the controlled unitary $C(R(\theta) ^ k)$. We can also illustrate here how to do some resource estimation with `qp.specs`.
    """)
    return


@app.cell
def _(qp, qpe):
    for n_bits in (2, 4, 8, 16):
        specs = qp.specs(qpe)(4, n_bits)
        print(f"{n_bits} bits")
        print("==============")
        print(specs)
        print()
    return


if __name__ == "__main__":
    app.run()
