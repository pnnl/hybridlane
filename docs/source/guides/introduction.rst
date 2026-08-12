Introduction
============

**hybridlane** implements heterogeneous quantum programming using PennyLane -- circuits composed of
both continuous-variable (CV) and discrete-variable (DV) degrees of freedom. We'll focus on qumodes (CV) and qubits (DV), but in principle hybridlane can handle qudits as well.

.. note::

    As hybridlane builds extensively on PennyLane, we highly recommend reading the `PennyLane documentation <https://docs.pennylane.ai/en/stable/>`_.

hybridlane circuits are just PennyLane circuits with our additional `type checking
<type-checking>`_ procedure. In PennyLane's common "quantum function" format, circuits are defined
in a functional manner. Each quantum function ``f(x)`` accepts a set of classical inputs ``x``,
invokes a series of quantum operations, and returns a set of classical outputs obtained by
measuring the quantum state after the operations.

Simple example
--------------

As a simple example, here's a circuit that produces a definite Fock state :math:`\ket{n}` by
repeatedly shuffling quanta from a qubit to a qumode:

.. code:: python

    import numpy as np
    import pennylane as qp
    import hybridlane as hl

    dev = qp.device('default.hybrid', fock_level=8)

    @qp.qnode(dev)
    def circuit(n):
        for j in range(n):
            qp.X("q")
            hl.JC(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, wires=["q", "m"])

        return hl.expval(hl.N("m"))

.. warning::

    For our experimentalist friends -- hybridlane uses the quantum information convention that
    :math:`Z\ket{0} = +1\ket{0}` is the initial (ground) state of the quantum processor. This may
    differ from other libraries like QuTiP and Dynamiqs, but it's to ensure consistency with
    PennyLane.

The input to the quantum function ``circuit`` is a single integer ``n``, determining which state to
prepare. Then, a sequence of interleaving :math:`X` and :math:`JC` gates are applied, and finally
the mean photon number of the qumode, :math:`\langle \hat{n} \rangle`, is returned. The circuit
can then be evaluated for different values of ``n``:

.. code-block:: python

    >>> circuit(5)
    5

.. note::

    You may have noticed the ``@qp.qnode(dev)`` decorator. This is required to bind the quantum
    function to a particular device -- in this case, the ``default.hybrid`` simulator. Without it,
    the function won't be executable.

Upon invoking the circuit, the ``default.hybrid`` device invokes hybridlane's type checker to
validate that the circuit is well-formed, and it determines which wires are qubits and qumodes by
inspecting the gates used in the circuit. The first gate encountered was the Pauli ``X`` gate,
which constrains wire ``q`` to be a qubit. Then, the next gate is a hybrid ``JC`` gate, which is
defined to act on a qubit and a qumode, in that order. So, the type checker determines that ``m``
is a qumode, and it also validates the previous assignment of ``q`` as a qubit.

You can check the results yourself using ``hl.type_check``, which returns a *function* that you
then call with the same arguments as the original quantum function:

.. code-block:: python

    >>> hl.type_check(circuit)(5)
    TypeCheckResult(wire_types=OrderedDict({'q': Qubit(), 'm': Qumode()}), basis_maps=[BasisMap({'m': ComputationalBasis.Discrete})])

In practice, you'll almost never need to do this.

.. tip::

    You can see the full list of qubit operations at :mod:`pennylane` and CV/hybrid operations
    at :mod:`hybridlane`.
