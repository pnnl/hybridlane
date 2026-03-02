# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

r"""Device definition for Sandia Qscout ion trap"""

import math
from collections.abc import Sequence
from dataclasses import replace
from functools import partial, singledispatch
from typing import Hashable, cast

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
    resolve_dynamic_wires,
    single_qubit_fusion,
)
from pennylane.wires import Wires

import hybridlane as hqml
from hybridlane.ops.op_math.decompositions import make_gate_with_ancilla_qubit

from ... import sa
from ...measurements import SampleMeasurement
from ...transforms import from_pennylane
from . import jaqal
from . import ops as native_ops
from .jaqal import Qumode

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
# Assign non-native CD gates a higher cost so it'll use the xCD gate
NATIVE_GATES_WITH_COST = {g: 1 for g in NATIVE_GATES} | {
    "ConditionalDisplacement": 3,
    "ConditionalYDisplacement": 3,
}


# Define constraints on the gates
@singledispatch
def is_gate_supported(op: Operator):
    return op.name in NATIVE_GATES


@is_gate_supported.register
def _(op: native_ops.ConditionalXSqueezing):
    # Hardcoded to the tilt mode on the lower manifold
    return op.wires[1] == Qumode(1, 1)


@is_gate_supported.register
def _(op: native_ops.NativeBeamsplitter):
    # Only supported between the tilt modes
    is_tilt_modes = op.wires.contains_wires(Wires([Qumode(0, 1), Qumode(1, 1)]))
    is_supported_qubit = op.wires[0] in Wires([0, 1, 3])  # zig zag indexing
    return is_tilt_modes and is_supported_qubit


@single_tape_support
class QscoutIonTrap(Device):
    r"""Backend that prepares circuits for the Sandia QSCOUT ion trap

    This device can't actually execute anything; instead, it's intended as a
    compilation target for the QSCOUT ion trap :footcite:p:`clark2021engineering`. As
    an example of how to compile a circuit to this device,

    .. code:: python

        dev = qml.device("sandiaqscout.hybrid")

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            hqml.FockState(5, [0, "m1i1"])
            return hqml.expval(qml.Z(0))

        tape = qml.workflow.construct_tape(circuit)()

    References
    ----------

    .. footbibliography::
    """

    name = "Sandia Qscout Ion Trap"  # type: ignore
    shortname = "qscout"
    version = "0.2.0"
    pennylane_requires = ">=0.44.0"
    author = "PNNL"

    _max_qubits = 6
    _device_options = (
        "n_qubits",
        "optimize",
        "enable_com_modes",
        "use_virtual_wires",
    )

    def __init__(
        self,
        wires: int | Sequence[Hashable] | None = None,
        shots: int | None = None,
        n_qubits: int | None = None,
        optimize: bool = True,
        enable_com_modes: bool = False,
        use_virtual_wires: bool = True,
    ):
        r"""Initializes the device

        Args:
            wires: An optional list of wires to expect in each circuit. If this is
                passed, then executing a circuit will error if it has any wire not in
                `wires`

            shots: The number of shots to use for a measurement

        Keyword arguments:
            See the options of :func:`get_compiler`
        """
        if n_qubits is not None and n_qubits > self._max_qubits:
            raise DeviceError(
                f"Requested more qubits than available "
                f"({n_qubits} > {self._max_qubits})"
            )

        if not use_virtual_wires:
            qubits = n_qubits or self._max_qubits
            wires = _get_allowed_device_wires(qubits, enable_com_modes)

        super().__init__(wires=wires, shots=shots)

        self._n_qubits = n_qubits
        self._optimize = optimize
        self._enable_com_modes = enable_com_modes
        self._use_virtual_wires = use_virtual_wires

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
    ) -> qml.CompilePipeline:
        execution_config = execution_config or ExecutionConfig()
        device_options = execution_config.device_options.copy() or {}
        device_options.setdefault("max_qubits", self._max_qubits)
        if n_qubits := device_options.pop("n_qubits", None):
            device_options["max_qubits"] = n_qubits
        return get_compiler(**device_options)


def get_compiler(
    optimize: bool = True,
    max_qubits: int | None = None,
    enable_com_modes: bool = False,
    use_virtual_wires: bool = True,
) -> qml.CompilePipeline:
    r"""Returns a compilation pipeline for QscoutIonTrap device

    Args:
        optimize: Whether to perform any simplifications of the circuit including
            cancelling inverse gates, merging consecutive rotations, and commuting
            controlled operators

        max_qubits: The number of qubits per circuit. If None (default), this will be
            inferred from each circuit. By setting this number to more qubits than
            are used in the circuit, this can grant access to additional qumodes.

        enable_com_modes: If True, the center-of-mass qumodes are enabled. As they are
            likely to be very noisy due to heating, they are disabled by default.

        use_virtual_wires: If True (default), the circuit may contain algorithmic
            (virtual) wires that will be mapped to physical wires by the compiler.
            If False, the circuit must contain only physical wires.
    """
    pipeline: qml.CompilePipeline = (
        from_pennylane
        + diagonalize_measurements
        + dynamic_gate_decompose(gate_set=NATIVE_GATES_WITH_COST, max_qubits=max_qubits)
    )

    # At this point, everything is a native instruction so we can perform virtual
    # wire layout if desired.
    pipeline += parse_hardware_wires
    if use_virtual_wires:
        pipeline += layout_wires(
            max_qubits=max_qubits,
            use_com_modes=enable_com_modes,
        )

    if optimize:
        pipeline += (
            commute_controlled
            + cancel_inverses
            + merge_rotations
            + single_qubit_fusion
            + decompose(gate_set=NATIVE_GATES_WITH_COST)
            + _simplify_transform
            + decompose(gate_set=NATIVE_GATES_WITH_COST)
        )

    pipeline += combine_global_phases

    # Finally, validate the circuit
    pipeline += get_validator(max_qubits or QscoutIonTrap._max_qubits, enable_com_modes)
    return pipeline


def get_validator(
    max_qubits: int, enable_com_modes: bool = False
) -> qml.CompilePipeline:
    r"""Returns a validation pipeline for QscoutIonTrap device"""
    physical_wires = _get_allowed_device_wires(max_qubits, enable_com_modes)

    return (
        validate_device_wires(wires=physical_wires, name=QscoutIonTrap.name)
        + validate_measurements(
            analytic_measurements=lambda *_: False,
            sample_measurements=accepted_sample_measurement,
        )
        + validate_gates_supported_on_hardware
    )


@qml.transform
def validate_gates_supported_on_hardware(tape: QuantumScript):
    for op in tape.operations:
        if not is_gate_supported(op):
            raise DeviceError(f"Operation {op} is not supported natively")

    def null_postprocessing(results):
        return results[0]

    return (tape,), null_postprocessing


@qml.transform
def dynamic_gate_decompose(
    tape: QuantumScript,
    sa_res: sa.StaticAnalysisResult | None = None,
    max_qubits: int | None = None,
    gate_set: set | dict | None = None,
):
    if sa_res is None:
        sa_res = sa.analyze(tape)

    gate_set = gate_set or NATIVE_GATES_WITH_COST

    remaining_work_wires = None
    if max_qubits is not None:
        n_qubits = len(sa_res.qubits)
        remaining_work_wires = max_qubits - n_qubits

    # Decompose into the target gate set allowing dynamic qubit allocation, then map
    # dynamic wires to virtual wires
    pipeline = decompose(
        gate_set=gate_set,
        alt_decomps=DECOMPS,
        num_work_wires=remaining_work_wires,
        minimize_work_wires=True,
    ) + resolve_dynamic_wires(
        zeroed=[f"virtual-qubit-{i}" for i in range(remaining_work_wires or 0)],
        allow_resets=False,  # no native reset instruction
    )

    return pipeline(tape)


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


@qml.transform
def parse_hardware_wires(tape: QuantumScript):
    wire_map = {w: w for w in tape.wires}
    for wire in tape.wires:
        if (
            isinstance(wire, str)
            and (new_wire := Qumode.try_from_string(wire)) is not None
        ):
            wire_map[wire] = new_wire

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

    # For any wires in the circuit that are already hardware labels, force them to be
    # assigned to themselves
    for w in sa_res.qubits & hw_qubits:
        problem.addConstraint(lambda assigned, w=w: assigned == w, [w])

    for w in sa_res.qumodes & hw_qumodes:
        problem.addConstraint(lambda assigned, w=w: assigned == w, [w])

    def constraint(*hw_wires, virtual_op: Operator):
        hw_op = virtual_op.map_wires(
            {w: w2 for w, w2 in zip(virtual_op.wires, hw_wires)}
        )
        return is_gate_supported(hw_op)

    # Add a constraint per gate that aligns with the conditions above
    for op in tape.operations:
        problem.addConstraint(partial(constraint, virtual_op=op), op.wires)

    return problem


# Define gate decompositions. Note that many gates have already been defined
# in pennylane in terms of R{x,y,z} gates, which are native.


@qml.register_resources({qml.IsingXX: 1, qml.RY: 2, qml.RX: 2})
def cnot_decomp(wires, **_):
    # Taken from https://en.wikipedia.org/wiki/Mølmer–Sørensen_gate#Description
    qml.RY(math.pi / 2, wires[0])
    qml.IsingXX(math.pi / 2, wires)
    qml.RX(-math.pi / 2, wires[1])
    qml.RX(-math.pi / 2, wires[0])
    qml.RY(-math.pi / 2, wires[0])


@qml.register_resources({qml.GlobalPhase: 1, native_ops.R: 2})
def rot_decomp(phi, theta, omega, wires, **_):
    native_ops.R(theta - math.pi, math.pi / 2 - phi, wires=wires)
    native_ops.R(math.pi, (omega - phi) / 2 + math.pi / 2, wires=wires)
    qml.GlobalPhase((phi + omega) / 2)


DYNAMIC_DECOMPS = {
    hqml.D: [make_gate_with_ancilla_qubit(hqml.D)],
    hqml.S: [make_gate_with_ancilla_qubit(hqml.S)],
    hqml.R: [make_gate_with_ancilla_qubit(hqml.R)],
}

DECOMPS = {
    qml.CNOT: [cnot_decomp],
    qml.Rot: [rot_decomp],
} | DYNAMIC_DECOMPS


def _get_allowed_device_wires(max_qubits: int, use_com_modes: bool) -> Wires:
    qubits = _get_device_qubits(max_qubits)
    qumodes = _get_device_qumodes(max_qubits, use_com_modes)
    return qubits + qumodes


def _get_device_qubits(max_qubits: int) -> Wires:
    return Wires(range(max_qubits))


def _get_device_qumodes(max_qubits: int, use_com_modes: bool) -> Wires:
    min_qumode_idx = 1 - use_com_modes
    qumodes = Wires(
        [Qumode(m, i) for i in range(min_qumode_idx, max_qubits) for m in (0, 1)]
    )
    return qumodes
