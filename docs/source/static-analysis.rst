Static Analysis
===============

Hybridlane infers the types of wires and measurements through a process we call static analysis. This is necessary because
Pennylane does not strongly type the wires (in contrast to other major libraries) - a wire can represent a qubit or a qumode. We
need types to verify that gates are used correctly, and to properly dispatch circuits to hardware and simulator devices. The process is fairly simple and is contained in the function :py:func:`sa.analyze <hybridlane.sa.analyze>`.

Inferring Wire Types
--------------------

The first way Hybridlane infers the type of wires is through gates.

1. Normal Pennylane gates (that subclass :py:class:`~pennylane.operation.Operation`) are assumed to be DV gates, meaning that all the input wires are qubits.

2. Pennylane and Hybridlane gates that subclass :py:class:`~pennylane.operation.CVOperation` are assumed to be CV gates, and therefore all input wires are qumodes.

3. Finally, for hybrid gates, Hybridlane provides the :py:class:`~hybridlane.ops.Hybrid` mixin. The ``Hybrid`` mixin enforces the convention that wires are in order ``[*qubits, *qumodes]``. As long as hybrid gates use that mixin and provide the ``num_qumodes`` attribute, the circuit checker has no issue inferring which wires are qubits and which are qumodes.

.. tip::
    Once a wire type is deduced from a gate, it is fixed for the remainder of the circuit, meaning if a subsequent gate treats a wire in a manner differently than its original definition, an error is thrown. We refer to this as *aliasing*, where a qubit is used as a qumode or vice versa.

Hybridlane can also infer types through the measurements performed on a wire, usually from measuring an observable. This follows a similar set of rules:

1. Simple DV observables that define ``pauli_rep`` are assumed to be DV observables. All input wires are then qubits.

2. Simple CV observables (subclassing :py:class:`~pennylane.operation.CVObservable`) are assumed to be CV observables. All input wires are then qumodes.

3. Complex observables (subclassing :py:class:`~pennylane.ops.op_math.CompositeOp` or :py:class:`~pennylane.ops.op_math.SymbolicOp`) like ``Sum``, ``Adjoint``, and ``Prod`` are traversed recursively, with each of their terms being analyzed using these rules.

.. tip::

    We don't use the ``Hybrid`` mixin for observables, as we assume that an observable can eventually be broken down into a term like :math:`O_Q \otimes O_B`, which would be represented in Pennylane as ``dv_obs @ cv_obs``.

Inferring Measurements
----------------------

The type of measurement required in a quantum program is captured by a combination of the wire type (qubit/qumode) and its :py:class:`~hybridlane.sa.BasisSchema`. Qubits are obviously measured in the computational basis (Z), which is represented by ``ComputationalBasis.Discrete``. Qumodes, however, have 3 measurement possibilities:

1. ``ComputationalBasis.Discrete`` in conjunction with a qumode means to apply a Fock measurement, sampling :math:`\hat{n}`. This is often called photon number readout. The result of this measurement should be stored in an ``int`` or ``uint`` type.

2. ``ComputationalBasis.Position`` measures a qumode in the position basis, sampling :math:`\hat{x}`. This is commonly referred to as homodyne detection. The result of this measurement requires a ``float`` type.

3. ``ComputationalBasis.Coherent`` measures the Husimi-Q function of a state, returning a coherent state :math:`\ket{\alpha}`. This is referred to as heterodyne detection. The result of this measurement requires a ``complex`` type.

For observables, the logic to deduce which measurement (contained in :py:func:`sa.infer_schema_from_observable <hybridlane.sa.infer_wires.infer_schema_from_observable>`) to apply works as follows:

1. Composite operators (subclassing :py:class:`~pennylane.ops.op_math.CompositeOp` or :py:class:`~pennylane.ops.op_math.SymbolicOp`) are recursively traversed.

2. If an operator defines ``pauli_rep``, it is assigned ``ComputationalBasis.Discrete`` because it must be a qubit observable.

3. For CV observables that implement the :py:class:`~hybridlane.ops.mixins.Spectral` mixin, we use that operator's ``natural_basis``. For observables that can be decomposed to :math:`\hat{n}`, this becomes ``ComputationalBasis.Discrete``, and for observables that can be decomposed to :math:`\hat{x}`, this becomes ``ComputationalBasis.Position``.

There's almost certainly room to improve this inference logic, so ideas and pull requests welcome!

.. tip::

    Up until now, we haven't mentioned what the ``Spectral`` mixin does. Pennylane assumes that all observables have a finite set of eigenvalues that can be written down in a numpy array. Obviously this doesn't work for CV observables that have an infinite eigenspectrum.
    
    The ``Spectral`` mixin replaces the idea of an ``eigvals`` array with a function :math:`f: \mathcal{B} \rightarrow \mathbb{R}` taking computational basis states and returning their eigenvalues. For example, we add the ``Spectral`` mixin to the :py:class:`hqml.QuadX <hybridlane.ops.cv.QuadX>` operator with its ``position_spectrum`` function looking like :math:`f(x) = x` since :math:`\hat{x}\ket{x} = x\ket{x}`. This is another reason to use the ``hqml`` versions when available.
