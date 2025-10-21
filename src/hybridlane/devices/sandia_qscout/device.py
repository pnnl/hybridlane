# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

r"""Device definition for Sandia Qscout ion trap"""

import math
from collections.abc import Sequence
from dataclasses import replace
from functools import partial, singledispatch
from typing import Hashable, cast

import numpy as np
import pennylane as qml
from pennylane.devices import Device
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.devices.modifiers import single_tape_support
from pennylane.devices.preprocess import (
    validate_device_wires,
    validate_measurements,
)
from pennylane.exceptions import DeviceError
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.ops.functions.simplify import _simplify_transform
from pennylane.pauli import PauliSentence
from pennylane.tape import QuantumScript
from pennylane.transforms import (
    cancel_inverses,
    combine_global_phases,
    commute_controlled,
    decompose,
    diagonalize_measurements,
    merge_rotations,
    single_qubit_fusion,
)
from pennylane.transforms.core import TransformProgram
from pennylane.wires import Wires

import hybridlane as hqml

from ... import sa
from ...measurements import SampleMeasurement
from ...transforms import from_pennylane
from ..preprocess import static_analyze_tape
from . import jaqal, ops
from .ops import (
    ConditionalXDisplacement,
)

# --------------------------------------------
#     Rules about what the device handles
# --------------------------------------------


def accepted_sample_measurement(m: MeasurementProcess) -> bool:
    if not isinstance(m, SampleMeasurement):
        return False

    if m.obs is not None:
        return is_sampled_observable_supported(m.obs)

    return True


def is_sampled_observable_supported(o: Operator) -> bool:
    if o.pauli_rep:
        pr = cast(PauliSentence, o.pauli_rep)
        return len(pr) == 1

    return False


NATIVE_GATES = set(jaqal.QUBIT_GATES) | set(jaqal.BOSON_GATES)


# Define constraints on the gates
@singledispatch
def is_gate_supported(_: Operator):
    return True


@is_gate_supported.register
def _(op: ops.FockStatePrep):
    # Hardcoded to the tilt modes
    return op.wires[1] in ("a0m1", "a1m1")


@is_gate_supported.register
def _(op: ops.ConditionalXDisplacement):
    # Hardcoded to the tilt mode on axis 0
    return op.wires[1] == "a0m1"


@is_gate_supported.register
def _(op: ops.ConditionalXSqueezing):
    # Hardcoded to the tilt mode on axis 0
    return op.wires[1] == "a0m1"


@is_gate_supported.register
def _(op: ops.NativeBeamsplitter):
    # Only supported between the tilt modes
    return op.wires.contains_wires(Wires(["a0m1", "a1m1"]))


@single_tape_support
class QscoutIonTrap(Device):
    r"""Backend for Pennylane that prepares circuits to be run on the Sandia QSCOUT ion trap

    This device can't actually execute anything; instead, it's intended as a compilation target.
    As an example of how to compile a circuit to this device,

    .. code:: python

        import pennylane as qml
        import hybridlane as hqml
        from hybridlane.devices.sandia_qscout import QscoutIonTrap
        from pennylane.workflow import construct_tape

        dev = QscoutIonTrap(shots=1000)

        @qml.qnode(dev)
        def circuit():
            hqml.FockLadder(5, [0, "a0m1"])
            return hqml.expval(qml.Z(0))

        tape = construct_tape(circuit)()

    """

    name = "Sandia Qscout Ion Trap"  # type: ignore
    shortname = "qscout"
    version = "0.1.0"
    pennylane_requires = ">=0.42.0"
    author = "PNNL"

    _max_qubits = 6
    _device_options = (
        "n_qubits",
        "optimize",
        "use_com_modes",
        "use_hardware_wires",
        "use_fockstate_instruction",
    )

    def __init__(
        self,
        wires: int | Sequence[Hashable] | None = None,
        shots: int | None = None,
        n_qubits: int | None = None,
        optimize: bool = True,
        use_com_modes: bool = False,
        use_hardware_wires: bool = False,
        use_fockstate_instruction: bool = True,
    ):
        r"""Initializes the device

        Args:
            wires: An optional list of wires to expect in each circuit. If this is passed, then executing
                a circuit will error if it has any wire not in `wires`

            shots: The number of shots to use for a measurement

            n_qubits: The number of qubits per circuit. If None (default), this will be inferred from
                each circuit. By setting this number to more qubits than are used in the circuit,
                this can grant access to additional qumodes.

            optimize: Whether to perform any simplifications of the circuit including cancelling inverse
                gates, merging consecutive rotations, and commuting controlled operators

            use_com_modes: If True, the center-of-mass qumodes are enabled. As they are likely
                to be very noisy due to heating, they are disabled by default.

            use_hardware_wires: If True, the circuit must contain only physical qumode/qubit wires.
                If False (default), the device will assign available physical qubits and qumodes to
                algorithmic (virtual) wires.

            use_fockstate_instruction: If True (default), all :class:`~hybridlane.FockLadder` gates
                acting on the tilt modes will be replaced by the native instruction instead of
                being decomposed into blue/red gates.
        """
        if n_qubits is not None and n_qubits > self._max_qubits:
            raise DeviceError(
                f"Requested more qubits than available ({n_qubits} > {self._max_qubits})"
            )

        if use_hardware_wires:
            qubits = n_qubits or self._max_qubits
            wires = _get_allowed_device_wires(qubits, use_com_modes)

        super().__init__(wires=wires, shots=shots)

        self._n_qubits = n_qubits
        self._optimize = optimize
        self._use_com_modes = use_com_modes
        self._use_hardware_wires = use_hardware_wires
        self._use_fockstate_instruction = use_fockstate_instruction

    def execute(  # type: ignore
        self,
        circuits: Sequence[QuantumScript],
        execution_config: ExecutionConfig | None = None,
    ):
        # We can't actually execute anything, instead this device is just meant
        # as a compilation target.
        return (0,) * len(circuits)

    def setup_execution_config(
        self,
        config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> ExecutionConfig:
        config = config or ExecutionConfig()
        updated_values = {}

        for option in config.device_options or {}:
            if option not in self._device_options:
                raise DeviceError(f"Device option {option} not present on {self}")

        updated_values["device_options"] = dict(config.device_options)  # copy

        for option in self._device_options:
            if option not in updated_values["device_options"]:
                updated_values["device_options"][option] = getattr(self, f"_{option}")

        if circuit and updated_values["device_options"].get("n_qubits") is None:
            sa_res = sa.analyze(circuit)
            updated_values["device_options"]["n_qubits"] = len(sa_res.qubits)

        return replace(config, **updated_values)

    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> TransformProgram:
        execution_config = execution_config or ExecutionConfig()
        device_options = execution_config.device_options or {}

        transform_program = TransformProgram()

        # Convert pennylane gates to hybridlane
        transform_program.add_transform(from_pennylane)

        # Make everything measurable in pauli z basis. Diagonalize prior to
        # decompose so that it decomposes unsupported gates
        transform_program.add_transform(diagonalize_measurements)

        # Decompose all gates into the native gate set. We include FockLadder so that
        # it can be laid out, and then after the layout stage we'll decompose it if necessary.
        transform_program.add_transform(
            decompose, gate_set=NATIVE_GATES | {"FockLadder"}
        )

        # If virtual wires are allowed, we need to assign them
        max_qubits = device_options.get("n_qubits", self._n_qubits)
        use_com_modes = device_options.get("use_com_modes", self._use_com_modes)
        if not device_options.get("use_hardware_wires", self._use_hardware_wires):
            transform_program.add_transform(
                layout_wires,
                max_qubits=max_qubits,
                use_com_modes=use_com_modes,
            )

        # Validate all wires are assigned to proper physical addresses
        allowed_wires = _get_allowed_device_wires(
            max_qubits or self._max_qubits, use_com_modes
        )
        transform_program.add_transform(
            validate_device_wires, wires=allowed_wires, name=self.name
        )

        # Qubit/qumode type checking
        transform_program.add_transform(static_analyze_tape)

        # Measurement check
        transform_program.add_transform(
            validate_measurements,
            analytic_measurements=lambda *_: False,
            sample_measurements=accepted_sample_measurement,
            name=self.name,
        )

        # Expand any unsupported fockstate operations because decompose transform
        # below won't (it's a native gate)
        transform_program.add_transform(
            map_supported_fockstate,
            use_native_instruction=device_options.get(
                "use_fockstate_instruction", self._use_fockstate_instruction
            ),
        )

        # Do one more transformation that now expands unsupported fock ladders
        transform_program.add_transform(decompose, gate_set=NATIVE_GATES)

        # Optional optimizations
        if device_options.get("optimize", self._optimize):
            transform_program.add_transform(commute_controlled)
            transform_program.add_transform(cancel_inverses)
            transform_program.add_transform(merge_rotations)

            # Fuse single qubit gates into Rot, then hopefully decompose them into
            # R gates
            transform_program.add_transform(single_qubit_fusion)
            transform_program.add_transform(decompose, gate_set=NATIVE_GATES)

            # Simplify any rotation parameters, then decompose again will delete
            # Identity from the circuit
            transform_program.add_transform(_simplify_transform)
            transform_program.add_transform(decompose, gate_set=NATIVE_GATES)

        transform_program.add_transform(combine_global_phases)

        # Finally check all native gates. This takes into account additional constraints
        # like gates being defined only on certain qumodes that decompose doesn't know about
        transform_program.add_transform(validate_gates_supported_on_hardware)

        return transform_program


@qml.transform
def validate_gates_supported_on_hardware(tape: QuantumScript):
    for op in tape.operations:
        if not is_gate_supported(op):
            raise DeviceError(f"Operation {op} is not supported natively")

    def null_postprocessing(results):
        return results[0]

    return (tape,), null_postprocessing


@qml.transform
def layout_wires(
    tape: QuantumScript,
    sa_res: sa.StaticAnalysisResult | None = None,
    max_qubits: int | None = None,
    use_com_modes: bool = False,
):
    if sa_res is None:
        sa_res = sa.analyze(tape)

    max_qubits = max_qubits or len(sa_res.qubits)
    max_qumodes = 2 * max_qubits if use_com_modes else 2 * max_qubits - 2
    qubits, qumodes = sa_res.qubits, sa_res.qumodes

    if len(qubits) > max_qubits:
        raise DeviceError(
            f"Circuit has more qubits ({len(qubits)}) than the maximum "
            f"requested or allowed ({max_qubits})"
        )

    if len(qumodes) > max_qumodes:
        raise DeviceError(
            f"Circuit has more qumodes ({len(qumodes)}) than the maximum "
            f"requested or allowed ({max_qumodes})"
        )

    wire_map = _constrained_layout(
        tape, sa_res, max_qubits=max_qubits, use_com_modes=use_com_modes
    )

    if wire_map is None:
        raise DeviceError(
            "No layout was found that could implement the gates in the circuit"
        )

    def null_postprocessing(results):
        return results[0]

    tape_batch, _ = qml.map_wires(tape, wire_map)
    return tape_batch, null_postprocessing


def _constrained_layout(
    tape: QuantumScript,
    sa_res: sa.StaticAnalysisResult,
    max_qubits: int | None = None,
    use_com_modes: bool = False,
) -> dict | None:
    max_qubits = max_qubits or len(sa_res.qubits)
    hw_qubits = _get_device_qubits(max_qubits)
    hw_qumodes = _get_device_qumodes(max_qubits, use_com_modes)

    # Todo: Possible improvement is to iterate through solutions and score
    # them based on qumode assignment/gate fidelities
    problem = _construct_csp(tape, sa_res, hw_qubits, hw_qumodes)
    wire_map = problem.getSolution()
    return wire_map


def _construct_csp(
    tape: QuantumScript,
    sa_res: sa.StaticAnalysisResult,
    hw_qubits: Wires,
    hw_qumodes: Wires,
):
    from constraint import AllDifferentConstraint, Problem

    # We'll solve the layout (note: not routing) as a constraint satisfaction problem.
    # The inputs are virtual wires, and our output is hardware wires. Each gate
    # potentially restricts the domain of a virtual wire's assignments.

    problem = Problem()

    # Ensure we get valid wire types out
    problem.addVariables(sa_res.qubits, hw_qubits)
    problem.addVariables(sa_res.qumodes, hw_qumodes)

    # All virtual wires must have unique hardware assignments
    problem.addConstraint(AllDifferentConstraint())

    def constraint(*hw_wires, virtual_op: Operator):
        data, (_, *hashable_hyperparameters) = virtual_op._flatten()  # pyright: ignore[reportPrivateUsage]
        hw_op = virtual_op._unflatten(
            data, (Wires(hw_wires), *hashable_hyperparameters)
        )
        return is_gate_supported(hw_op)

    # Add a constraint per gate that aligns with the conditions above
    for op in tape.operations:
        problem.addConstraint(partial(constraint, virtual_op=op), op.wires)

    return problem


@qml.transform
def map_supported_fockstate(tape: QuantumScript, use_native_instruction: bool = True):
    new_ops = []

    for op in tape.operations:
        if isinstance(op, hqml.FockLadder):
            native_op = ops.FockStatePrep(*op.parameters, wires=op.wires)
            if use_native_instruction and is_gate_supported(native_op):
                new_ops.append(native_op)

            else:
                new_ops.append(op)

        else:
            new_ops.append(op)

    def null_postprocessing(results):
        return results[0]

    new_tape = QuantumScript(new_ops, tape.measurements, tape.shots)
    return (new_tape,), null_postprocessing


# Define gate decompositions. Note that many gates have already been defined
# in pennylane in terms of R{x,y,z} gates, which are native.


@qml.register_resources({qml.IsingXX: 1, qml.RY: 2, qml.RX: 2})
def _cnot_decomp(wires, **_):
    # Taken from https://en.wikipedia.org/wiki/Mølmer–Sørensen_gate#Description
    qml.RY(math.pi / 2, wires[0])
    qml.IsingXX(math.pi / 2, wires)
    qml.RX(-math.pi / 2, wires[1])
    qml.RX(-math.pi / 2, wires[0])
    qml.RY(-math.pi / 2, wires[0])


qml.add_decomps(qml.CNOT, _cnot_decomp)


@qml.register_resources({qml.GlobalPhase: 1, ops.R: 2})
def _rot_decomp(phi, theta, omega, wires, **_):
    ops.R(theta - math.pi, math.pi / 2 - phi, wires=wires)
    ops.R(math.pi, (omega - phi) / 2 + math.pi / 2, wires=wires)
    qml.GlobalPhase((phi + omega) / 2)


qml.add_decomps(qml.Rot, _rot_decomp)


@qml.register_resources({ConditionalXDisplacement: 1, qml.H: 2})
def _conditionaldisplacement_decomp(*params, wires, **_):
    r, phi = params
    beta = r * np.exp(1j * phi)

    qml.H(wires[0])
    ConditionalXDisplacement(np.real(beta), np.imag(beta), wires=wires)
    qml.H(wires[0])


qml.add_decomps(hqml.ConditionalDisplacement, _conditionaldisplacement_decomp)

# Currently unknown how to decompose normal squeezing parameters to red/blue sideband ratios
# @register_resources({ConditionalXSqueezing: 1, qml.H: 2})
# def _conditionalsqueezing_decomp(*params, wires, **_):
#     qml.H(wires[0])
#     ConditionalXSqueezing(*params, wires=wires)
#     qml.H(wires[0])


# add_decomps(hqml.ConditionalSqueezing, _conditionalsqueezing_decomp)


def _get_allowed_device_wires(max_qubits: int, use_com_modes: bool) -> Wires:
    qubits = _get_device_qubits(max_qubits)
    qumodes = _get_device_qumodes(max_qubits, use_com_modes)
    return qubits + qumodes


def _get_device_qubits(max_qubits: int) -> Wires:
    return Wires(range(max_qubits))


def _get_device_qumodes(max_qubits: int, use_com_modes: bool) -> Wires:
    min_qumode_idx = 1 - use_com_modes
    qumodes = Wires(
        [f"a{a}m{i}" for i in range(min_qumode_idx, max_qubits) for a in (0, 1)]
    )
    return qumodes
