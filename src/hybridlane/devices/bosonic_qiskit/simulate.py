# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from __future__ import annotations

import functools
import math
import warnings
from typing import Callable

import bosonic_qiskit as bq
import numpy as np
import pennylane as qml
from pennylane.exceptions import DeviceError
from pennylane.operation import Operator
from pennylane.ops import Exp, Pow, Prod, SProd, Sum
from pennylane.ops.cv import CVOperation
from pennylane.tape import QuantumScript
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from qiskit.primitives import BitArray
from qiskit.quantum_info import Statevector
from qiskit.result import Result as QiskitResult
from scipy import sparse as sp
from scipy.linalg import expm, fractional_matrix_power
from scipy.sparse import SparseEfficiencyWarning, csc_array

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
bq.operators.SMINUS[:] = bq.operators.SMINUS.T
bq.operators.SPLUS[:] = bq.operators.SPLUS.T


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


def analytic_state(
    state: Statevector,
    result: QiskitResult,
    obs: np.ndarray,
    regmapper: RegisterMapping,
) -> np.ndarray:
    source_wires = Wires(
        range(len(regmapper.wire_order))
    )  # auto pulling from ALL hybridlane wires
    destination_wires = (
        regmapper.wire_order
    )  # how those wires map to the bq statevector wires
    state_size = state.data.shape[0]

    out_vector = -1 * np.ones(state.data.shape, dtype=complex)

    order = permute_subsystems(
        sp.diags([range(state_size)], [0]),  # matrix
        source_wires,  # 'observable' wires
        destination_wires,  # 'statevector' wires
        regmapper,
        qiskit_order=True,
    ).diagonal()
    for i, idx in enumerate(order):
        out_vector[int(idx)] = state.data[i]

    assert not np.any(out_vector == -1)
    return out_vector


analytic_measurement_map: dict[
    type[SampleMeasurement],
    Callable[[Statevector, QiskitResult, np.ndarray], np.ndarray],
] = {
    ExpectationMP: analytic_expval,
    VarianceMP: analytic_var,
    ProbabilityMP: analytic_probs,
    # StateMP: analytic_state,  # Don't even need StateMP if the device handles the calculation
}


def get_sparse_observable_matrix(
    obs: Operator, *cutoffs: int, hbar: float
) -> csc_array:
    if not cutoffs:
        raise ValueError("Expected at least one cutoff")

    lam = np.sqrt(hbar / 2)
    cvops = bq.operators.CVOperators()

    def get_x(c: int):
        return lam * (cvops.get_a(c) + cvops.get_ad(c))

    def get_p(c: int):
        return lam * -1j * (cvops.get_a(c) - cvops.get_ad(c))

    match obs:
        case qml.Identity():
            return cvops.get_eye(cutoffs[0])

        case hqml.NumberOperator():
            return cvops.get_N(cutoffs[0])

        case hqml.QuadX():
            return get_x(cutoffs[0])

        case hqml.QuadP():
            return get_p(cutoffs[0])

        case hqml.QuadOperator(parameters=(phi,)):
            return np.cos(phi) * get_x(cutoffs[0]) + np.sin(phi) * get_p(cutoffs[0])

        case hqml.FockStateProjector(parameters=(fock_states,)):
            mats = [cvops.get_projector(n, c) for n, c in zip(fock_states, cutoffs)]
            return functools.reduce(sp.kron, mats).asformat("csc")

        case _:
            mat = (
                obs.sparse_matrix(format="csc")
                if obs.has_sparse_matrix
                else obs.matrix()
            )
            return csc_array(mat)


def get_observable_matrix(
    obs: Operator, regmapper: RegisterMapping, *, hbar: float
) -> np.ndarray:
    # Here we need to construct the matrix for the observable in the wire order
    # expected by qiskit.

    if not obs.is_verified_hermitian:
        raise DeviceError(f"Got non-hermitian observable {obs}")

    # Handle symbolic observable expressions by traversing the expression tree
    match obs:
        case Sum(operands=terms):
            return sum(get_observable_matrix(o, regmapper, hbar=hbar) for o in terms)
        case SProd(base=op, scalar=scalar):
            return scalar * get_observable_matrix(op, regmapper, hbar=hbar)
        case Exp(base=op, scalar=scalar):
            return expm(scalar * get_observable_matrix(op, regmapper, hbar=hbar))
        case Pow(base=op, scalar=pow):
            mat = get_observable_matrix(op, regmapper, hbar=hbar)
            try:
                return np.linalg.matrix_power(mat, pow)
            except TypeError:  # non-integer power
                return fractional_matrix_power(mat, pow)
        case Prod(operands=ops):
            if not util.is_tensor_product(obs):
                mats = map(
                    lambda x: get_observable_matrix(x, regmapper, hbar=hbar), ops
                )
                return functools.reduce(lambda x, y: x @ y, mats)

    # If we make it here, we should have a simple operator or a tensor product
    # We need to construct the observable matrix for each individual operator, then
    # expand the tensor product in the wire order defined by regmapper.wires to produce a
    # matrix that acts on the full state vector
    op_list = obs.operands if isinstance(obs, Prod) else (obs,)

    # Get matrices for component operators. Each component should act on disjoint wires
    op_mats: list[sp.csc_array] = []
    for op in op_list:
        cutoffs = tuple(map(regmapper.truncation.dim, op.wires))
        mat = get_sparse_observable_matrix(op, *cutoffs, hbar=hbar)
        op_mats.append(mat)

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

    if method := dv_gate_map.get(op.name):
        qubits = [regmapper.get(w) for w in wires]

        match op:
            # This is equivalent up to a global phase of e^{-i(φ + ω)/2}
            case qml.Rot(parameters=(phi, theta, omega)):
                getattr(qc, method)(
                    theta, phi, omega, *qubits
                )  # note the reordered parameters
            case _:
                getattr(qc, method)(*op.parameters, *qubits)

    elif isinstance(op, CVOperation) and (method := cv_gate_map.get(op.name)):
        qumodes = [regmapper.get(w) for w in wires]

        match op:
            # These gates take complex parameters or differ from bosonic qiskit
            case hqml.Displacement(parameters=(r, phi)):
                arg = r * np.exp(1j * phi)
                getattr(qc, method)(arg, *qumodes)
            case hqml.Squeezing(parameters=(r, phi)):
                arg = -r * np.exp(-1j * phi)
                getattr(qc, method)(arg, *qumodes)
            case hqml.Rotation(parameters=(theta,)):
                getattr(qc, method)(-theta, *qumodes)
            case hqml.Beamsplitter(parameters=(theta, phi)):
                new_theta = theta / 2
                new_phi = phi - np.pi / 2
                z = new_theta * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes)
            case hqml.TwoModeSqueezing(parameters=(r, phi)):
                new_phi = phi + np.pi / 2
                z = r * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes)
            case _:
                getattr(qc, method)(*op.parameters, *qumodes)

    elif isinstance(op, Hybrid) and (method := hybrid_gate_map.get(op.name)):
        wire_types = op.wire_types()

        qumodes = [regmapper.get(w) for w in op.wires if wire_types[w] == sa.Qumode()]
        qubits = [regmapper.get(w) for w in op.wires if wire_types[w] == sa.Qubit()]

        match op:
            case hqml.ConditionalRotation(parameters=(theta,)):
                getattr(qc, method)(-theta / 2, *qumodes, *qubits)
            case (
                hqml.ConditionalDisplacement(parameters=(r, phi))
                | hqml.ConditionalSqueezing(parameters=(r, phi))
            ):
                arg = r * np.exp(1j * phi)
                getattr(qc, method)(arg, *qumodes, *qubits)
            case (
                hqml.SQR(parameters=parameters, hyperparameters={"n": n})
                | hqml.SNAP(parameters=parameters, hyperparameters={"n": n})
            ):
                getattr(qc, method)(*parameters, n, *qumodes, *qubits)
            case hqml.ConditionalBeamsplitter(parameters=(theta, phi)):
                new_theta = theta / 2
                new_phi = phi - np.pi / 2
                z = new_theta * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes)
            case hqml.ConditionalTwoModeSqueezing(parameters=(r, phi)):
                new_phi = phi + np.pi / 2
                z = r * np.exp(1j * new_phi)
                getattr(qc, method)(z, *qumodes, *qubits)
            case _:
                getattr(qc, method)(*op.parameters, *qumodes, *qubits)

    elif method := misc_gate_map.get(op.name):
        match op:
            case qml.Barrier():
                pass  # no-op

    else:
        raise DeviceError(f"Unsupported operation {op.name}")


# todo: write unit tests for this function
def permute_subsystems(
    A: sp.csc_array,
    source_wires: Wires,
    destination_wires: Wires,
    regmapper: RegisterMapping,
    qiskit_order: bool = False,
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
    return (
        analytic_measurement_map.get(type(m))(state, result, obs)
        if type(m) in analytic_measurement_map
        else analytic_state(state, result, obs, regmapper)
    )


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
        if wire in regmapper.sa_res.qumodes:
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
