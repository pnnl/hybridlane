Introduction
============

Welcome to Hybridlane! Hybridlane is a frontend library for expressing CV-DV quantum circuits, quantum programs that use both qubits and qumodes. Being a frontend library, we focus on providing tools to abstractly describe those circuits independently of the hardware or simulation, and leave the computational heavy-lifting to other libraries.

In this introduction, we'll briefly cover what Hybridlane does and how to use it with simple examples. These examples will be familiar to Pennylane users, as our library builds on top of it, so we highly recommend reading through the Pennylane documentation as well.

... CV-DV?
----------

Hybridlane implements the so-called "continuous-variable/discrete-variable" (CV-DV) paradigm described by Liu *et al* (`paper <https://arxiv.org/abs/2407.10381>`_). A CV-DV circuit has both discrete variable (DV) components (qubits) and continuous-variable (CV) components (qumodes). As a quick review, the qubits have 2 discrete states, :math:`\{\ket{0}, \ket{1}\}`, while the qumodes have an infinite number of states corresponding to the quantum harmonic oscillator. Unlike the qubit case, qumodes have two equivalent computational bases: the **position basis** (:math:`\{\ket{x}, x\in\mathbb{R}\}`) and the **Fock basis** (:math:`\{\ket{n}, n \in \mathbb{N}_0\}`). These are linked by the relation

.. math::

    \psi(x) = \sum_{n=0}^\infty \psi_n \braket{x|n}

where :math:`\braket{x|n} = \phi_n(x)` are the quantum harmonic oscillator eigenstates.

An example
----------

Defining a quantum circuit in Hybridlane follows the same format as Pennylane, and to illustrate this, we'll show a sample circuit and then walk through the steps.

.. code:: python

    import pennylane as qml
    import hybridlane as hqml

    dev = qml.device('hybrid.device')

    @qml.qnode(dev)
    def circuit(n):
        for j in range(n):
            qml.X(0)
            hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [1, 0])
        
        return hqml.expval(hqml.NumberOperator(1))
    
    n = 5
    result = circuit(n)
    
This circuit consists of one qubit (wire ``0``) and one qumode (wire ``1``), and it prepares the state :math:`\ket{0, n}` by (:math:`n` times) pumping one quanta into the qubit and transferring it into the qumode.

.. note::

    In Hybridlane, each circuit starts in the state :math:`\ket{0\dots 0}`. Differing from the paper, we use the convention that qubits start in their ground state, that is :math:`\ket{0} = \ket{g} = \ket{\uparrow}` and :math:`\ket{1} = \ket{e} = \ket{\downarrow}`. This maintains the conventions used by Pennylane and other mainstream quantum libraries.

Let's walk through the steps:

.. code:: python

    import pennylane as qml
    import hybridlane as hqml

This code imports both Pennylane (``qml``) to get access to its decorators and their quantum gates, and our library Hybridlane (``hqml``). Usually you will need both - our library uses all the existing qubit and qumode gates provided by Pennylane, along with extra hybrid (qubit-qumode) gates and utilities provided by us. Pay close attention to which parts use Pennylane functions (``qml.``) and Hybridlane functions (``hqml.``).

.. code:: python

    dev = qml.device('hybrid.device')

When simulating circuits in Pennylane, circuits are usually bound to devices. You must choose a device that supports the operations in your circuit, or you'll obtain an error (for example, using ``qml.Displacement`` gates with the ``default.qubit`` device). This line initializes the device registered with name ``hybrid.device``. Note that while we picked a fictional device for the purposes of providing a clean example, Hybridlane does provide a reference simulator device based on Bosonic Qiskit, ``hybrid.bosonicqiskit``.

.. code:: python

    @qml.qnode(dev)
    def circuit(n):
        for j in range(n):
            qml.X(0)
            hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [1, 0])
        
        return hqml.expval(hqml.NumberOperator(1))

This middle part is the actual circuit definition consisting of its inputs, operation, and outputs (measurements). The special decorator at the top, ``@qml.qnode(dev)``, is a Pennylane method for converting pure Python functions into executable quantum circuits. This binds the function ``circuit`` to the device ``dev``.

.. code:: python

    def circuit(n):
        for j in range(n):
            qml.X(0)
            hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [1, 0])

Next is our circuit definition, which accepts a single parameter :math:`n`. The function produces the circuit

.. math::

    \ket{\psi} = \left[\prod_{j=n-1}^{0} JC_{1, 0}\left(\frac{\pi}{2\sqrt{j+1}}, \frac{\pi}{2}\right) X_0 \right] \ket{0,0}.

Within a circuit definition, you are free to use Python control flow, like loops (``for``, ``while``) and conditionals (``if``, ``else``, ``match``). This makes circuits in Pennylane rather flexible. The last argument of a gate (e.g. ``qml.X`` or ``hqml.JaynesCummings``) is always the "wires", the qubit(s) and/or qumode(s) that the gate acts on, in order.

- The Pauli :math:`X` gate (``qml.X``) acts on a single qubit and so it accepts a single wire (qubit ``0``)

- The hybrid qubit-qumode gate :math:`JC(\theta, \phi)` (``hqml.JaynesCummings``) accepts two parameters and then acts on two wires (qumode ``1`` and qubit ``0``).

In Hybridlane we use the convention that all qumodes are listed before qubits (more on that later).

.. tip::

    You can find the list of gates provided by Pennylane at :py:mod:`pennylane`. The extra hybrid gates implemented by Hybridlane are at :py:mod:`hybridlane`. Again, if a class or function is provided under both ``qml`` and ``hqml`` (e.g. ``NumberOperator``, ``QuadX``, ``expval``), use the ``hqml`` version.

.. code:: python

    return hqml.expval(hqml.NumberOperator(1))

Finally, the return statement determines what measurements our circuit will make. In this case, we obtain the expectation value of the photon number operator on the qumode, :math:`\braket{\hat{n}_1}`. Here, we use the Hybridlane functions ``hqml.expval`` and ``hqml.NumberOperator``. Pennylane has its own versions, but we had to redefine them for additional functionality, so use our versions.

Phew that was a lot. But, up until this point (the last two lines), nothing has happened - these are just the *definitions*. Nothing actually happens until the function ``circuit`` is invoked,

.. code:: python

    n = 5
    result = circuit(n)

These lines pass the parameter :math:`n = 5` to our circuit, meaning we will prepare the state :math:`\ket{0, 5}` and measure its photon number :math:`\braket{\hat{n}_1} = 5`. Behind the scenes, Pennylane records the operations in our circuit definition, constructs a ``QuantumTape`` object, sends it to the device ``hybrid.device``, and returns the result (that's all the "magic" hidden behind the ``@qml.qnode`` decorator).

.. tip::

    At this point you might be wondering how Hybridlane determines which wires are qumodes and qubits in a circuit. The short answer is by inspecting the circuit structure and gate definitions, e.g. the wires of a qumode gate are inferred to be qumodes. This is why all our gates enforce the convention that qumodes come before qubits. Inference of the circuit structure is covered more in-depth in the :doc:`static-analysis` section.

Drawing the circuit
-------------------

Pennylane provides some utility methods for visualizing circuits, :py:func:`pennylane.draw` and :py:func:`pennylane.draw_mpl`, which (mostly) work on Hybridlane circuits. To view a textual representation of the circuit, we can do

.. code:: python

    print(qml.draw(circuit)(n))

which produces the output

.. code::

    0: ──X─╭JaynesCummings(1.57,1.57)──X─╭JaynesCummings(1.11,1.57)──X─╭JaynesCummings(0.91,1.57)──X ···
    1: ────╰JaynesCummings(1.57,1.57)────╰JaynesCummings(1.11,1.57)────╰JaynesCummings(0.91,1.57)─── ···

    0: ··· ─╭JaynesCummings(0.79,1.57)──X─╭JaynesCummings(0.70,1.57)─┤               
    1: ··· ─╰JaynesCummings(0.79,1.57)────╰JaynesCummings(0.70,1.57)─┤  expval(n̂(1))

A prettier graphic can be made through matplotlib, with

.. code:: python

    qml.draw_mpl(circuit, decimals=2, style='sketch')(5)

.. image:: _static/images/ex_jc_circuit.png
