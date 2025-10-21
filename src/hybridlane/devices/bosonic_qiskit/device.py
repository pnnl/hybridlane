# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

# A note on imports: This bosonic qiskit device is considered optional, and the user
# can choose whether or not to install support for it. However, pennylane will always
# load this file to register the device from the entrypoints in `pyproject.toml`. Therefore,
# this file needs to be importable *even if* bosonic qiskit is not installed.
#
# To facilitate this, the special dependencies of this device are only imported under the
# typing.TYPE_CHECKING statement or in the functions that actually need the packages. Currently,
# those dependencies are
#   - bosonic qiskit (c2qa)
#   - qiskit
#   - sparse (not the scipy.sparse package)
# + any other files in this directory (hybridlane.devices.bosonic_qiskit)


import importlib.util
from collections.abc import Sequence
from dataclasses import replace
from typing import Hashable

from pennylane.devices import Device
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.devices.modifiers import single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    validate_device_wires,
    validate_measurements,
)
from pennylane.exceptions import DeviceError
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.ops import CompositeOp, Prod, SymbolicOp
from pennylane.tape import QuantumScript
from pennylane.transforms.core import TransformProgram

from ... import sa, util
from ...measurements import (
    CountsMP,
    FockTruncation,
    SampleMeasurement,
    StateMeasurement,
)
from ...ops import attributes
from ...sa import ComputationalBasis
from ...transforms import from_pennylane
from ..preprocess import static_analyze_tape
from . import gates

# --------------------------------------------
#     Rules about what our device handles
# --------------------------------------------


def accepted_analytic_measurement(m: MeasurementProcess) -> bool:
    if not isinstance(m, StateMeasurement):
        return False

    if m.obs is not None:
        return is_analytic_observable_supported(m.obs)

    return True


def accepted_sample_measurement(m: MeasurementProcess) -> bool:
    if not isinstance(m, SampleMeasurement) or isinstance(m, CountsMP):
        return False

    if m.obs is not None:
        return is_sampled_observable_supported(m.obs)

    # If we're directly sampling the basis states, it has to be in Fock space
    for wire in m.schema.wires:
        if m.schema.get_basis(wire) != ComputationalBasis.Discrete:
            return False

    return True


def is_analytic_observable_supported(o: Operator) -> bool:
    # Fixme: there might be edge cases, but for the most part we've got
    # methods to derive the appropriate matrix for o. See `_get_observable_matrix`
    return True


def is_sampled_observable_supported(o: Operator) -> bool:
    match o:
        case SymbolicOp(base=base_op):
            return is_sampled_observable_supported(base_op)
        case Prod(operands=ops):
            # Only simple tensor products supported
            if not util.is_tensor_product(o):
                return False

            return all(is_sampled_observable_supported(op) for op in ops)
        case CompositeOp():
            # No strategy for more complicated composite operators like Sum
            return False
        case _:
            # We can only sample computational basis, so if the observable
            # prefers the position basis, we can't do that. The schema class
            # should be able to handle finite eigenvalues vs spectra, etc.
            schema = sa.infer_schema_from_observable(o)
            for wire in schema.wires:
                if schema.get_basis(wire) != ComputationalBasis.Discrete:
                    return False

            # We must also be able to diagonalize the observable
            return o.has_diagonalizing_gates or o in attributes.diagonal_in_fock_basis


# todo: (roadmap) add @simulator_tracking and enable resource tracking for QRE
@single_tape_support
class BosonicQiskitDevice(Device):
    r"""Backend for Pennylane that executes hybrid CV-DV circuits in Bosonic Qiskit"""

    name = "Bosonic Qiskit"  # type: ignore
    shortname = "bosonic-qiskit"
    version = "0.1.0"
    author = "PNNL"

    _device_options = ("truncation", "hbar")

    def __init__(
        self,
        wires: Sequence[Hashable] | None = None,
        shots: int | None = None,
        max_fock_level: int | None = None,
        truncation: FockTruncation | None = None,
        hbar: float = 2,
    ):
        r"""Initializes the device

        Args:
            wires: An optional list of wires to expect in each circuit. If this is passed, then executing
                a circuit will error if it has any wire not in `wires`

            shots: The number of shots to use for a measurement. If `None` (the default), this performs
                analytic measurements

            max_fock_level: The cutoff to apply uniformly across all qumodes.

            truncation: An optional truncation that allows for more granular cutoffs specified per-qumode.
                This must be passed if `max_fock_level` is None.

            hbar: The value for the constant :math:`\bar{h}`.
        """

        if importlib.util.find_spec("c2qa") is None:
            raise ImportError(
                f"The {self.name} device depends on bosonic-qiskit, "
                "which can be installed with `pip install hybrid-circuit[bq]`"
            )

        if truncation and not isinstance(truncation, FockTruncation):
            raise DeviceError(f"Only Fock truncations are supported, got {truncation}")

        self._truncation = truncation
        self._max_fock_level = max_fock_level
        self._hbar = hbar

        super().__init__(wires=wires, shots=shots)

    def execute(  # type: ignore
        self,
        circuits: Sequence[QuantumScript],
        execution_config: ExecutionConfig | None = None,
    ):
        from .simulate import simulate

        execution_config = execution_config or ExecutionConfig()
        truncation = execution_config.device_options.get("truncation", self._truncation)
        max_fock_level = execution_config.device_options.get(
            "max_fock_level", self._max_fock_level
        )

        # Try to infer truncation based on circuit structure
        if truncation is None:
            sa_results = map(sa.analyze, circuits)
            truncations = list(
                map(lambda res: _infer_truncation(res, max_fock_level), sa_results)
            )
            if any(t is None for t in truncations):
                raise DeviceError(
                    "Unable to infer truncation for one of the circuits in the batch. Need to specify truncation "
                    "of qumodes through `device_options`"
                )
        else:
            truncations = [truncation] * len(circuits)

        return tuple(
            simulate(tape, truncation, hbar=self._hbar)
            for tape, truncation in zip(circuits, truncations)
        )

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

        # If there is no truncation at the device level or in this particular config, try to
        # auto-generate one if the circuit is purely qubits
        if (
            updated_values["device_options"].get("truncation") is None
            and circuit is not None
        ):
            max_fock_level = updated_values["device_options"].get(
                "max_fock_level", self._max_fock_level
            )
            res = sa.analyze(circuit)
            if (truncation := _infer_truncation(res, max_fock_level)) is None:
                raise DeviceError(
                    "Need to specify truncation of qumodes through `device_options`"
                )
            updated_values["device_options"]["truncation"] = truncation

        return replace(config, **updated_values)

    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> TransformProgram:
        execution_config = execution_config or ExecutionConfig()
        transform_program = TransformProgram()

        # Check that all wires aren't abstract
        transform_program.add_transform(validate_device_wires, name=self.name)

        # Convert pennylane gates to hybridlane
        transform_program.add_transform(from_pennylane)

        # Qubit/qumode type checking
        transform_program.add_transform(static_analyze_tape)

        # Todo: Add measurement decomposition

        # Measurement check
        transform_program.add_transform(
            validate_measurements,
            analytic_measurements=accepted_analytic_measurement,
            sample_measurements=accepted_sample_measurement,
            name=self.name,
        )

        transform_program.add_transform(
            decompose,
            stopping_condition=lambda o: type(o) in gates.supported_operations,
        )

        return transform_program


def _infer_truncation(
    sa_result: sa.StaticAnalysisResult, max_fock_level: int | None
) -> FockTruncation | None:
    if sa_result.qumodes and max_fock_level is None:
        return None

    qumodes = {w: max_fock_level for w in sa_result.qumodes}
    qubits = {w: 2 for w in sa_result.qubits}

    truncation = FockTruncation.all_fock_space(sa_result.wire_order, qumodes | qubits)
    return truncation
