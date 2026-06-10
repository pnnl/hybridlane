# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from itertools import starmap
from typing import Any, cast

import numpy as np
import pennylane as qp
from pennylane import CompilePipeline
from pennylane.concurrency.executors import RemoteExec, get_executor
from pennylane.decomposition import GateSet
from pennylane.devices.default_qubit import (
    _BASE_DQ_GATE_SET,
    ALL_DQ_GATES,
)
from pennylane.devices.device_api import Device, ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    device_resolve_dynamic_wires,
    no_sampling,
    validate_device_wires,
    validate_measurements,
)
from pennylane.devices.qubit.sampling import jax_random_split
from pennylane.exceptions import DeviceError
from pennylane.gradients.parameter_shift import param_shift
from pennylane.gradients.parameter_shift_cv import param_shift_cv
from pennylane.logging import debug_logger, debug_logger_init
from pennylane.math import Interface
from pennylane.measurements.measurements import MeasurementProcess
from pennylane.operation import Operation, Operator
from pennylane.ops.mid_measure.measurement_value import MeasurementValue
from pennylane.ops.op_math.composite import CompositeOp
from pennylane.ops.op_math.linear_combination import LinearCombination
from pennylane.ops.op_math.sum import Sum
from pennylane.ops.op_math.symbolicop import SymbolicOp
from pennylane.tape import QuantumScript
from pennylane.tape.qscript import QuantumScriptOrBatch
from pennylane.transforms.convert_to_numpy_parameters import convert_to_numpy_parameters
from pennylane.transforms.decompose import decompose
from pennylane.transforms.defer_measurements import defer_measurements
from pennylane.transforms.dynamic_one_shot import dynamic_one_shot
from pennylane.typing import PostprocessingFn, Result, ResultBatch
from pennylane.wires import WiresLike

import hybridlane as hl
from hybridlane.devices.default_hybrid.measure import is_diagonalizable
from hybridlane.devices.preprocess import fill_wire_dims
from hybridlane.measurements import (
    ComputationalBasis,
    ExpectationMP,
    SampleMeasurement,
    SampleMP,
    StateMeasurement,
)
from hybridlane.ops.mixins import FockRepresentation
from hybridlane.transforms import from_pennylane

from ... import math, sa
from .simulate import simulate

_base_qubit_gates = _BASE_DQ_GATE_SET

_base_cv_gates = {
    "Beamsplitter",
    "CubicPhase",
    "Displacement",
    "Fourier",
    "Kerr",
    "ModeSwap",
    "Rotation",
    "SelectiveNumberArbitraryPhase",
    "Squeezing",
    "TwoModeSqueezing",
    "TwoModeSum",
}

_base_hybrid_gates = {
    "AntiJaynesCummings",
    "ConditionalBeamsplitter",
    "ConditionalDisplacement",
    "ConditionalParity",
    "ConditionalRotation",
    "ConditionalSqueezing",
    "ConditionalTwoModeSqueezing",
    "ConditionalTwoModeSum",
    "EchoedConditionalDisplacement",
    "JaynesCummings",
    "SelectiveQubitRotation",
}

_state_preps = {
    "CatState",
    "FockState",
    "GaussianState",
    "SqueezedState",
    "CoherentState",
    "FockStateVector",
    "BasisState",
    "StatePrep",
}

ALL_DH_GATES = GateSet(
    ALL_DQ_GATES
    | _base_cv_gates
    | {f"Adjoint({g})" for g in _base_cv_gates}
    | {f"C({g})" for g in _base_cv_gates}
    | _base_hybrid_gates
    | {f"Adjoint({g})" for g in _base_hybrid_gates}
    | _state_preps,
    name="All DefaultHybrid gates",
)

ALL_DH_GATES_PLUS_MCM = GateSet(
    ALL_DH_GATES | {"MidMeasureMP"}, name="All DefaultHybrid gates + MCM"
)


def stopping_condition(op: Operator, allow_mcm: bool = True) -> bool:
    from pennylane.devices.default_qubit import (
        stopping_condition as dq_stopping_condition,
    )

    if isinstance(op, Operation):
        return op in ALL_DH_GATES_PLUS_MCM if allow_mcm else op in ALL_DH_GATES

    return dq_stopping_condition(op, allow_mcms=allow_mcm)


stopping_condition_no_mcm = partial(stopping_condition, allow_mcm=False)
stopping_condition_with_mcm = partial(stopping_condition, allow_mcm=True)


def is_analytic_mp_supported(mp: MeasurementProcess) -> bool:
    if not isinstance(mp, StateMeasurement):
        return False

    if mp.obs is not None:
        return is_analytic_observable_supported(mp.obs)

    return True


def is_analytic_observable_supported(obs: Operator | MeasurementValue) -> bool:
    match obs:
        case SymbolicOp(base=base_op):
            return is_analytic_observable_supported(base_op)
        case CompositeOp(operands=ops):
            return all(map(is_analytic_observable_supported, ops))
        case MeasurementValue():
            return True

    return (
        obs.has_matrix or obs.has_sparse_matrix or isinstance(obs, FockRepresentation)
    )


def is_sampled_mp_supported(mp: MeasurementProcess) -> bool:
    if not isinstance(mp, SampleMeasurement):
        return False

    if mp.obs is not None:
        return is_sampled_observable_supported(
            mp.obs, is_expval=isinstance(mp, ExpectationMP)
        )

    # hl.sample() called with a schema
    return all(mp.schema.get_basis(w) == ComputationalBasis.Discrete for w in mp.wires)


def is_sampled_observable_supported(
    obs: Operator | MeasurementValue, is_expval: bool
) -> bool:
    match obs:
        case MeasurementValue():
            return True

        case Sum(operands=ops) | LinearCombination(operands=ops):
            if is_expval:
                return all(is_diagonalizable(op) for op in ops)

            return len(ops) == 1 and is_diagonalizable(ops[0], is_expval)

        case _:
            return is_diagonalizable(obs)


@simulator_tracking
@single_tape_support
class DefaultHybrid(Device):
    """A hybridlane device written in Python capable of backpropagation"""

    name = "default.hybrid"
    author = "PNNL"
    version = hl.__version__
    pennylane_requires = ">=0.45.0"

    _device_options = ("fock_level", "wire_dims", "max_workers", "rng", "prng_key")

    @debug_logger_init
    def __init__(
        self,
        fock_level: int | None = None,
        wire_dims: Mapping[Any, int] | None = None,
        wires: WiresLike | None = None,
        seed: Any = "global",
        shots: int | None = None,
        max_workers: int | None = None,
    ):
        super().__init__(wires=wires, shots=shots)
        seed = np.random.randint(0, 2**32) if seed == "global" else seed
        if math.get_interface(seed) == "jax":
            self._prng_seed = self._prng_key = seed
            self._rng = np.random.default_rng(None)
        else:
            self._prng_seed = self._prng_key = None
            self._rng = np.random.default_rng(seed)

        self._debugger = None
        self._fock_level = fock_level
        self._wire_dims = wire_dims
        self._max_workers = max_workers

    @debug_logger
    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        if execution_config is None:
            return True

        no_max_workers = (
            execution_config.device_options.get("max_workers", self._max_workers)
            is None
        )

        if execution_config.gradient_method in {"backprop", "best"} and no_max_workers:
            if circuit is None:
                return True

            return not circuit.shots  # backprop incompatible with sampling

        # no adjoint support
        return execution_config.gradient_method in {param_shift, param_shift_cv, None}

    @debug_logger
    def setup_execution_config(
        self,
        config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> ExecutionConfig:
        config = config or ExecutionConfig()
        updated_values = {}

        if not qp.capture.enabled():
            # This logic comes from default.qubit, and it captures the following logic:
            # - If the user passes a prng_key, they obviously intend to use jax
            # - Adjoint differentiation requires caching in default.qubit, which
            #   is incompatible with jax
            # - Higher order derivatives also seem to require caching
            updated_values["convert_to_numpy"] = not (
                self._prng_key is not None
                and config.interface in {Interface.JAX, Interface.JAX_JIT}
                and config.gradient_method != "adjoint"
                and config.derivative_order == 1
            )

        gradient_method = config.gradient_method
        if gradient_method not in {
            "backprop",
            "best",
            param_shift,
            param_shift_cv,
            None,
        }:
            raise DeviceError(
                f"Gradient method '{gradient_method}' is not supported by {self.name}."
            )

        if config.use_device_gradient is None:
            updated_values["use_device_gradient"] = gradient_method in {
                "backprop",
                "best",
            }

        # ----- Device options -----
        for option in config.device_options:
            if option not in self._device_options:
                raise DeviceError(
                    f"Device option '{option}' is not supported by {self.name}."
                )

        updated_values["device_options"] = dict(config.device_options)
        default_device_options = {
            k: getattr(self, f"_{k}") for k in self._device_options
        }
        updated_values["device_options"] = (
            default_device_options | updated_values["device_options"]
        )

        # Check the truncations. Only one may be specified because if `fock_level` is
        # specified, we will use the same cutoff across all qumodes (whose wires will be
        # determined in execute()). Perhaps we should consider renaming `wire_dims` to
        # `overrides` and allowing it to provide non-default cutoffs and infer everything
        # else.
        wire_dims = cast(
            Mapping[Any, int] | None, updated_values["device_options"]["wire_dims"]
        )
        fock_level = cast(int | None, updated_values["device_options"]["fock_level"])
        if (wire_dims is not None) == (fock_level is not None):
            raise DeviceError(
                "Exactly one of 'wire_dims' or 'fock_level' must be specified."
            )

        return replace(config, **updated_values)

    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> CompilePipeline:
        config = execution_config or ExecutionConfig()
        pipeline = CompilePipeline()
        target_gate_set = ALL_DH_GATES

        if config.interface == Interface.JAX_JIT:
            pipeline.add_transform(no_sample)

        match config.mcm_config.mcm_method:  # ty:ignore[unresolved-attribute]
            case "deferred":
                pipeline.add_transform(defer_measurements, allow_postselect=True)
                stopping_condition = stopping_condition_no_mcm
                allow_resets = False
            case None:
                stopping_condition = stopping_condition_no_mcm
                allow_resets = False
            case _:
                target_gate_set = ALL_DH_GATES_PLUS_MCM
                stopping_condition = stopping_condition_with_mcm
                allow_resets = True

                # todo: remove in v0.9.0
                raise DeviceError("Mid-circuit measurement isn't supported")

        # Convert PennyLane gates and measurements prior to decomposition
        pipeline.add_transform(from_pennylane)

        # todo: remove in v0.9.0
        pipeline.add_transform(_batching_is_unsupported)

        # fixme: whenever we can change the device version of this transform to
        # use our decompositions, then switch this
        pipeline.add_transform(
            decompose,
            gate_set=target_gate_set,
            stopping_condition=stopping_condition,
        )
        pipeline.add_transform(
            device_resolve_dynamic_wires, allow_resets=allow_resets, wires=self.wires
        )
        pipeline.add_transform(
            fill_wire_dims,
            wire_dims=config.device_options.get("wire_dims", self._wire_dims),
            default_qumode_dim=config.device_options.get(
                "fock_level", self._fock_level
            ),
        )
        pipeline.add_transform(validate_device_wires, self.wires, name=self.name)
        pipeline.add_transform(
            validate_measurements,
            analytic_measurements=is_analytic_mp_supported,
            sample_measurements=is_sampled_mp_supported,
            name=self.name,
        )

        if config.mcm_config.mcm_method == "one-shot":  # ty:ignore[unresolved-attribute]
            pipeline.add_transform(
                dynamic_one_shot,
                postselect_mode=config.mcm_config.postselect_mode,  # ty:ignore[unresolved-attribute]
            )

        if config.gradient_method == "backprop":
            pipeline.add_transform(no_sampling, name=f"backprop + {self.name}")

        return pipeline

    @debug_logger
    def execute(
        self,
        circuits: Sequence[QuantumScript],
        execution_config: ExecutionConfig | None = None,
    ) -> ResultBatch:
        if execution_config is None:
            execution_config = ExecutionConfig()

        max_workers = execution_config.device_options.get("max_workers")
        self._prng_key = self._prng_seed
        self._prng_key, *prng_keys = jax_random_split(self._prng_key, len(circuits) + 1)

        # Get the concrete wire_dims by performing type inference
        wire_dims = list(map(lambda t: _get_wire_dims(t, execution_config), circuits))
        remapped_circuits = map(QuantumScript.map_to_standard_wires, circuits)

        if max_workers is None:
            kwargs = (
                {
                    "rng": self._rng,
                    "prng_key": prng_key,
                    "interface": execution_config.interface,
                    "debugger": self._debugger,
                }
                for prng_key in prng_keys
            )
            return tuple(starmap(_simulator, zip(remapped_circuits, wire_dims, kwargs)))

        remapped_circuits = tuple(
            map(lambda tape: convert_to_numpy_parameters(tape)[0][0], remapped_circuits)
        )
        rngs = self._rng.integers(2**31 - 1, size=len(circuits))
        kwargs = (
            {
                "rng": rng,
                "prng_key": prng_key,
                "interface": execution_config.interface,
                "debugger": self._debugger,
            }
            for rng, prng_key in zip(rngs, prng_keys)
        )

        assert execution_config.executor_backend is not None
        backend = get_executor(execution_config.executor_backend)  # ty:ignore[invalid-argument-type]
        with backend(max_workers=max_workers) as executor:
            executor = cast(RemoteExec, executor)
            results = tuple(
                executor.map(_simulator, remapped_circuits, wire_dims, kwargs)
            )

        self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))
        return results


def _simulator(tape: QuantumScript, wire_dims: dict[int, int], kwargs) -> Result:
    return simulate(
        tape,
        wire_dims=wire_dims,
        **kwargs,
    )


def _get_wire_dims(tape: QuantumScript, config: ExecutionConfig) -> dict[int, int]:
    """Helper function to obtain the wire_dims for a tape

    This must be called *before* remapping the tape's wires or we won't be able to remap
    the wire dimensions as well


    Returns:
        The wire dimensions mapped to standard wire order as determined by
            ``tape._get_standard_wire_map()``
    """

    # Guaranteed in setup_execution_config that exactly one of these is not None
    wire_dims = config.device_options.get("wire_dims")
    fock_level = cast(int | None, config.device_options.get("fock_level"))

    # If the user provided a blanket value for all qumodes, we have to construct the
    # wire_dims by type checking the circuit
    if fock_level is not None:
        res = sa.analyze(tape)
        wire_dims = {w: 2 for w in res.qubits} | {w: fock_level for w in res.qumodes}

    # Guaranteed because we just overrode it above if it was None
    wire_dims = cast(Mapping[Any, int], wire_dims)
    if wire_map := tape._get_standard_wire_map():
        wire_dims = {wire_map.get(w, w): d for w, d in wire_dims.items()}

    return wire_dims  # ty:ignore[invalid-return-type]


# todo: remove in v0.9.0
@qp.transform
def _batching_is_unsupported(
    tape: QuantumScript,
) -> tuple[QuantumScriptOrBatch, PostprocessingFn]:
    for op in tape.operations:
        if op.batch_size:
            raise DeviceError(
                "Operator batching is not supported, but operation"
                f" {op} has batch size {op.batch_size}. Consider using `jax.vmap`"
            )

    return (tape,), lambda x: x[0]


@qp.transform
def no_sample(tape: QuantumScript) -> tuple[QuantumScriptOrBatch, PostprocessingFn]:
    for mp in tape.measurements:
        if isinstance(mp, SampleMP):
            raise DeviceError(f"`jax.jit` does not support {mp}")

    return (tape,), lambda x: x[0]
