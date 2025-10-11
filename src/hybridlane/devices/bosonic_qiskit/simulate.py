# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from __future__ import annotations

import functools
import math
import warnings
from typing import Callable

import c2qa as bq
import c2qa.operators
import numpy as np
import pennylane as qml
from pennylane.exceptions import DeviceError
from pennylane.operation import Operator
from pennylane.ops import Pow, Prod, SProd, Sum
from pennylane.ops.cv import CVObservable, CVOperation
from pennylane.tape import QuantumScript
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from qiskit.primitives import BitArray
from qiskit.quantum_info import Statevector
from qiskit.result import Result as QiskitResult
from scipy import sparse as sp
from scipy.sparse import SparseEfficiencyWarning, csc_matrix

import hybridlane as hqml

from ... import sa, util
from ...measurements import (
    ExpectationMP,
    FockTruncation,
    ProbabilityMP,
    SampleMeasurement,
    SampleResult,
    StateMeasurement,
    VarianceMP,
)
from ...ops.mixins import Hybrid
from .gates import cv_gate_map, dv_gate_map, hybrid_gate_map, misc_gate_map
from .register_mapping import RegisterMapping

# Patch to flip the conventions from |g> = |1>, |e> = |0> to |g> = |0>, |e> = |1>
c2qa.operators.sigma_minus[:] = c2qa.operators.sigma_minus.T
c2qa.operators.sigma_plus[:] = c2qa.operators.sigma_plus.T


def simulate(
    tape: QuantumScript, truncation: FockTruncation, *, hbar: float
) -> tuple[np.ndarray]:
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

    qc, regmapper = make_cv_circuit(tape, truncation)

    if tape.shots and not len(tape.shots.shot_vector) == 1:
        raise NotImplementedError("Complex shot batching is not yet supported")

    results = []

    # Sampled measurements
    if tape.shots:
        for m in tape.measurements:
            assert isinstance(m, SampleMeasurement)

            exec_qc = qc.copy()  # reuse base circuit
            shots = tape.shots.total_shots
            sample_result = sampled_measurement(m, exec_qc, regmapper, shots)
            results.append(m.process_samples(sample_result, m.wires))

    # Analytic measurements
    else:
        # Compute state once and reuse across measurements to reduce simulation time
        state, result, _ = bq.util.simulate(qc, shots=None, return_fockcounts=False)
        for m in tape.measurements:
            assert isinstance(m, StateMeasurement)
            results.append(analytic_measurement(m, state, result, regmapper, hbar=hbar))

    if len(tape.measurements) == 1:
        return results[0]

    return tuple(results)


def analytic_expval(
    state: Statevector, result: QiskitResult, obs: np.ndarray
) -> np.ndarray:
    from qiskit.quantum_info import Operator

    return np.array(state.expectation_value(Operator(obs)).real)


def analytic_var(
    state: Statevector, result: QiskitResult, obs: np.ndarray
) -> np.ndarray:
    from qiskit.quantum_info import Operator

    op = Operator(obs)
    var = state.expectation_value(op**2) - state.expectation_value(op) ** 2
    return np.array(var.real)


def analytic_probs(
    state: Statevector, result: QiskitResult, obs: np.ndarray | None = None
) -> np.ndarray:
    # todo: somehow we need to take the statevector of 2^{num_qubits} and reshape/process it to
    # have shape (d1, ..., dn) with di being the dimension of system i. Then we'll also need to
    # move the wires around to match the original quantumtape/basis schema wire ordering

    # probs = state.probabilities()

    raise NotImplementedError()


analytic_measurement_map: dict[
    type[SampleMeasurement],
    Callable[[Statevector, QiskitResult, np.ndarray], np.ndarray],
] = {
    ExpectationMP: analytic_expval,
    VarianceMP: analytic_var,
    ProbabilityMP: analytic_probs,
}


def get_truncated_matrix_generator(
    obs: Operator, *, hbar: float
) -> Callable[[int], csc_matrix]:
    from c2qa.operators import CVOperators

    lam = np.sqrt(hbar / 2)
    cvops = CVOperators()

    def get_x(c):
        return lam * (cvops.get_a(c) + cvops.get_a_dag(c))

    def get_p(c):
        return lam * -1j * (cvops.get_a(c) - cvops.get_a_dag(c))

    match obs:
        case qml.Identity():
            return lambda cutoff: sp.eye(cutoff, format="csc")  # type: ignore

        case hqml.NumberOperator():
            return cvops.get_N

        case hqml.QuadX():
            return get_x

        case hqml.QuadP():
            return get_p

        case hqml.QuadOperator(parameters=params):
            phi = params[0]
            return lambda c: np.cos(phi) * get_x(c) + np.sin(phi) * get_p(c)

        case hqml.FockStateProjector(parameters=params, wires=wires):
            if len(wires) > 1:
                raise DeviceError(
                    "Only support obtaining matrix for single-wire fock projectors"
                )

            n: int = params[0].item()
            return lambda cutoff: csc_matrix(
                ([1.0], ([n], [n])), shape=(cutoff, cutoff)
            )

        case _:
            if obs.has_sparse_matrix:
                return lambda _: obs.sparse_matrix(format="csc")  # type: ignore
            else:
                return lambda _: csc_matrix(obs.matrix())


def get_observable_matrix(
    obs: Operator, regmapper: RegisterMapping, *, hbar: float
) -> np.ndarray:
    # Here we need to construct the matrix for the observable in the wire order
    # expected by qiskit.
    if isinstance(obs, Sum):
        return sum(
            c * get_observable_matrix(o, regmapper, hbar=hbar)
            for c, o in zip(*obs.terms())
        )
    elif isinstance(obs, SProd):
        return obs.scalar * get_observable_matrix(obs.base, regmapper, hbar=hbar)
    elif isinstance(obs, Pow):
        return np.linalg.matrix_power(
            get_observable_matrix(obs.base, regmapper, hbar=hbar), obs.scalar
        )
    elif isinstance(obs, Prod) and not util.is_tensor_product(obs):
        # Todo: this will require unit tests to make sure the matmul order is correct
        mats = map(
            lambda x: get_observable_matrix(x, regmapper, hbar=hbar), obs.operands
        )
        return functools.reduce(lambda x, y: x @ y, mats)

    # Decompose fock state projectors on multiple qumodes as a proper product
    elif isinstance(obs, hqml.FockStateProjector) and len(obs.wires) > 1:
        new_obs = qml.prod(
            *[
                hqml.FockStateProjector(n, w)
                for n, w in zip(obs.parameters[0], obs.wires)
            ]
        )
        return get_observable_matrix(new_obs, regmapper, hbar=hbar)

    # If we make it here, we should have a simple operator or a tensor product
    # We need to construct the observable matrix for each individual operator, then
    # expand the tensor product in the wire order defined by regmapper.wires to produce a
    # matrix that acts on the full state vector
    op_list = obs.operands if isinstance(obs, Prod) else [obs]

    # Get matrices for component operators. Each component should be multi-qubit or one qumode - we assume
    # prior to this point that circuit transforms reduced multi-qumode or hybrid observables into simpler
    # observables.
    op_mats: list[sp.csc_array] = []
    for op in op_list:
        mat_func = get_truncated_matrix_generator(op, hbar=hbar)

        cutoff = -1  # dummy value for dv observables
        if isinstance(op, (CVObservable, Hybrid)):
            if len(op.wires) > 1:
                raise DeviceError(
                    "This device currently does not support general CV or Hybrid observables acting on >=2 wires. "
                    f"Perhaps your observable was not decomposed using circuit transforms? Got: {op}"
                )

            cutoff = regmapper.truncation.dim(op.wires[0])

        op_mats.append(
            mat_func(cutoff)  # type: ignore
        )

    # Get wire dimensions
    statevector_wires = regmapper.wire_order
    obs_wires = Wires.all_wires([o.wires for o in op_list])

    # Find the Hilbert dimension of the remaining (identity) operators and add an I gate
    if remaining_wires := statevector_wires - obs_wires:
        dims = regmapper.truncation.shape(remaining_wires)
        dim = np.prod(dims)
        op_mats.append(sp.eye_array(dim, format="csc"))  # type: ignore
        obs_wires = Wires.all_wires([obs_wires, remaining_wires])

    # Perform full tensor product and reorder the subsystem wires from those in op_list to the statevector wires
    composite_matrix = functools.reduce(sp.kron, op_mats)
    composite_matrix = permute_subsystems(
        composite_matrix,
        obs_wires,
        statevector_wires,
        regmapper,
        qiskit_order=True,
    )

    return composite_matrix.todense()


def make_cv_circuit(
    tape: QuantumScript, truncation: FockTruncation
) -> tuple[bq.CVCircuit, RegisterMapping]:
    res = sa.analyze(tape)
    regmapper = RegisterMapping(res, truncation)
    for wire, dim in regmapper.truncation.dim_sizes.items():
        if not (qubits := math.log2(dim)).is_integer():
            raise DeviceError(
                f"Only Fock powers of 2 are currently supported on this device, got {dim} on wire {wire} (log2: {qubits})"
            )

    try:
        qc = bq.CVCircuit(*regmapper.regs)
    except ValueError as e:
        raise DeviceError(
            "Bosonic qiskit currently doesn't support executing circuits without a qumode."
        ) from e

    for op in tape.operations:
        # Validate that we have actual values in the parameters
        for p in op.parameters:
            if qml.math.is_abstract(p):
                raise DeviceError(
                    "Need instantiated tensors to convert to qiskit. Circuit may contain Jax or TensorFlow tracing tensors."
                )

        apply_gate(qc, regmapper, op)

    return qc, regmapper


def apply_gate(qc: bq.CVCircuit, regmapper: RegisterMapping, op: Operator):
    wires = op.wires
    parameters = tuple(map(to_scalar, op.parameters))

    if method := dv_gate_map.get(type(op)):
        qubits = [regmapper.get(w) for w in wires]

        match type(op):
            # This is equivalent up to a global phase of e^{-i(φ + ω)/2}
            case qml.Rot:
                phi, theta, omega = parameters
                getattr(qc, method)(
                    theta, phi, omega, *qubits
                )  # note the reordered parameters
            case _:
                getattr(qc, method)(*parameters, *qubits)

    elif isinstance(op, CVOperation) and (method := cv_gate_map.get(type(op))):
        qumodes = [regmapper.get(w) for w in wires]

        match type(op):
            # These gates take complex parameters or differ from bosonic qiskit
            case hqml.Displacement:
                a, phi = parameters
                alpha = a * np.exp(1j * phi)
                getattr(qc, method)(alpha, *qumodes)
            case hqml.Rotation:
                theta = parameters[0]
                getattr(qc, method)(-theta, *qumodes)
            case hqml.Squeezing:
                r, phi = parameters
                z = r * np.exp(1j * phi)
                getattr(qc, method)(z, *qumodes)
            case hqml.Beamsplitter:
                theta, phi = parameters
                new_theta = theta / 2
                new_phi = phi - np.pi / 2
                z = new_theta * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes)
            case hqml.TwoModeSqueezing:
                r, phi = parameters
                new_phi = phi + np.pi / 2
                z = r * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes)
            case _:
                getattr(qc, method)(*parameters, *qumodes)

    elif isinstance(op, Hybrid) and (method := hybrid_gate_map.get(type(op))):
        qubits, qumodes = op.split_wires()
        qumodes = [regmapper.get(w) for w in qumodes]
        qubits = [regmapper.get(w) for w in qubits]

        match type(op):
            case hqml.ConditionalRotation:
                theta = parameters[0]
                getattr(qc, method)(-theta / 2, *qumodes, *qubits)
            case hqml.ConditionalDisplacement:
                a, phi = parameters
                alpha = a * np.exp(1j * phi)
                getattr(qc, method)(alpha, *qumodes, *qubits)
            case hqml.ConditionalSqueezing:
                z, phi = parameters
                zeta = z * np.exp(1j * phi)
                getattr(qc, method)(zeta, *qumodes, *qubits)
            case hqml.SelectiveQubitRotation:
                n: int = op.hyperparameters["n"]
                getattr(qc, method)(*parameters, n, *qumodes, *qubits)
            case hqml.SelectiveNumberArbitraryPhase:
                n: int = op.hyperparameters["n"]
                getattr(qc, method)(*parameters, n, *qumodes, *qubits)
            case hqml.ConditionalBeamsplitter:
                theta, phi = parameters
                new_theta = theta / 2
                new_phi = phi - np.pi / 2
                z = new_theta * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes)
            case hqml.ConditionalTwoModeSqueezing:
                r, phi = parameters
                new_phi = phi + np.pi / 2
                z = r * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes, *qubits)
            case _:
                getattr(qc, method)(*parameters, *qumodes, *qubits)

    elif method := misc_gate_map.get(type(op)):
        match op:
            case qml.Barrier():
                pass  # no-op

    else:
        raise DeviceError(f"Unsupported operation {type(op)}")


# todo: write unit tests for this function
def permute_subsystems(
    A: sp.csc_array,
    source_wires: Wires,
    destination_wires: Wires,
    regmapper: RegisterMapping,
    qiskit_order=False,
) -> sp.csc_array:
    # Dedicated sparse library that allows for proper nd-arrays unlike scipy.sparse
    import sparse

    # We reverse the destination to match qiskit little endian ordering.
    if qiskit_order:
        destination_wires = destination_wires[::-1]

    if source_wires == destination_wires:
        return A

    # Get the order of the input and output axes, which will allow us to
    # compute the appropriate permutation
    source_oaxes = tuple(range(len(source_wires)))
    dest_oaxes = tuple(destination_wires.indices(source_wires))

    # Here we identify the permutation from x -> y
    n = len(source_oaxes)
    source_axes = source_oaxes + tuple(i + n for i in source_oaxes)
    dest_axes = dest_oaxes + tuple(i + n for i in dest_oaxes)
    perm = tuple(map(int, np.argsort(dest_axes)[np.argsort(source_axes)]))

    # convert to sparse library for axes permutation
    # Reshape the operator from (d, d) to (o1, ..., on, i1, ..., in) where oi == ii
    hilbert_dim: int = A.shape[0]
    source_dims = regmapper.truncation.shape(source_wires)
    coo_A = sparse.COO.from_scipy_sparse(A)
    coo_A = coo_A.reshape(2 * source_dims)  # first #wires are output
    coo_A = coo_A.transpose(perm)

    # Convert back to regular matrix shape and scipy format
    coo_A = coo_A.reshape((hilbert_dim, hilbert_dim))
    return coo_A.tocsc()


def analytic_measurement(
    m: StateMeasurement,
    state: Statevector,
    result: QiskitResult,
    regmapper: RegisterMapping,
    *,
    hbar: float,
):
    obs = (
        get_observable_matrix(m.obs, regmapper, hbar=hbar)
        if m.obs is not None
        else None
    )
    return analytic_measurement_map.get(type(m))(state, result, obs)


def sampled_measurement(
    m: SampleMeasurement,
    qc: bq.CVCircuit,
    regmapper: RegisterMapping,
    shots: int,
) -> SampleResult:
    import qiskit as qk
    from qiskit_aer.primitives import SamplerV2 as Sampler

    # If we're sampling an observable then we need to diagonalize it
    if m.obs is not None and not m.samples_computational_basis:
        for op in m.diagonalizing_gates():
            apply_gate(qc, regmapper, op)

    qc.measure_all()

    # Use the sampler here because it's better geared towards finite samples than the usual qiskit result
    sampler = Sampler(default_shots=shots)
    pm = qk.generate_preset_pass_manager(backend=sampler._backend)
    isa_qc = pm.run(qc)
    job = sampler.run([isa_qc])
    result = job.result()[0]
    qiskit_samples: BitArray = next(
        iter(result.data.values())
    )  # there should only be one classicalregister

    basis_states = {}
    for wire, qubits in regmapper.mapping.items():
        if wire not in m.wires:
            continue

        # Qumode, convert back to fock space
        if isinstance(qubits, list):
            indices: list[int] = qc.get_qubit_indices(qubits)
            bitstrings = qiskit_samples.slice_bits(indices)
            factor = 2 ** np.arange(bitstrings.num_bits, dtype=int)

            # The use of order "little" here means the bits are in order (1, 2, 4, ...)
            data = bitstrings.to_bool_array(order="little")
            fock_values = np.sum(data * factor, axis=-1).reshape(shots)
            basis_states[wire] = fock_values.astype(
                np.uint32
            )  # this should be sufficient width

        # Qubit, just grab the relevant values
        else:
            index = qc.get_qubit_index(qubits)

            if index is None:
                raise RuntimeError(
                    "Not sure how we got here, couldn't locate qubit in circuit"
                )

            bitstrings = qiskit_samples.slice_bits(index)
            basis_states[wire] = bitstrings.array.reshape(shots)

    sample_result = SampleResult(basis_states)
    return sample_result


def to_scalar(tensor_like: TensorLike):
    if isinstance(tensor_like, (int, float, complex)):
        return tensor_like

    # For PennyLane tensors (qml.numpy.ndarray, tf.Tensor, torch.Tensor, jax.numpy.ndarray)
    # qml.numpy.asarray handles the conversion to a standard NumPy array for all interfaces.
    try:
        np_array = qml.numpy.asarray(tensor_like)
    except Exception as e:
        raise TypeError(
            f"Could not convert input to a NumPy array. Original error: {e}"
        )

    # Check if the array is indeed a scalar
    if np_array.shape != ():
        raise ValueError(
            f"Input tensor is not a scalar. Has shape {np_array.shape}. "
            "Only scalar tensors can be converted to a Python scalar using this function."
        )

    # Use .item() to extract the scalar value from a 0-dimensional NumPy array
    return np_array.item()
