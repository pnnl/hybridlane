# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import copy
import functools
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Hashable, Self

import pennylane as qp
from pennylane.measurements import (
    MeasurementProcess,
    MeasurementShapeError,
)
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pydantic import BaseModel, ConfigDict, Field, model_validator

import hybridlane as hl
import hybridlane.sa as sa  # fixes a circular import

# -----------------------------------
#           Sampling Methods
# -----------------------------------


@dataclass(frozen=True)
class SampleResult:
    r"""Container for the results of a sample-based measurement

    This class maps wires to arrays of the corresponding type required to contain the result (see
    :py:attr:`.ComputationalBasis.return_type`).
    """

    data: dict[Any, TensorLike]
    """Mapping from each wire to the basis states drawn from it

    Each array has shape ``(shots,)`` or an optional batch dimension ``(B, shots)``
    """

    schema: sa.BasisSchema

    def __post_init__(self):
        self.validate_sample_tensors(self.schema, self.data)

    @classmethod
    def from_basis_states(cls, basis_states: dict[Any, TensorLike]) -> Self:
        schema = sa.infer_schema_from_tensors(basis_states)
        return cls(data=basis_states, schema=schema)

    @functools.cached_property
    def shape(self) -> tuple[int, ...]:
        return hl.math.shape(next(iter(self.data.values())))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def batch_size(self) -> int | None:
        if self.ndim == 1:
            return None

        return self.shape[0]

    @property
    def shots(self) -> int:
        return self.shape[-1]

    def concatenate(self, other: Self) -> "SampleResult":
        if self.schema != other.schema:
            raise ValueError("Schemas of each result must match")

        if self.batch_size != other.batch_size:
            raise ValueError("Results must have the same outer dimension")

        new_tensors = {}
        for w in self.data.keys():
            new_tensors[w] = qp.math.concatenate([self.data[w], other.data[w]], axis=-1)

        return SampleResult(data=new_tensors, schema=self.schema)

    def slice(self, indices: slice):
        new_tensors = {}
        for w in self.data.keys():
            new_tensors[w] = self.data[w][..., indices]  # ty:ignore[not-subscriptable, invalid-argument-type]

        return SampleResult(data=new_tensors, schema=self.schema)

    @staticmethod
    def validate_sample_tensors(
        schema: sa.BasisSchema, tensors: dict[Hashable, TensorLike]
    ):
        r"""Validates the tensors with several checks

        This function tests:
            - The wires match between the schema and the data
            - The data type of each tensor matches what's expected from the schema
            - All wire data has the same shape

        Args:
            tensors: The set of wire-tensor pairs to check

        Raises:
            :py:class:`ValueError`: if any of the tensors don't have the expected data type,
            or if a wire is present in the schema but not in the tensors, or if a wire is
            present in the tensors but not the schema
        """
        wires = Wires(tensors.keys())
        if unexpected_wires := Wires.unique_wires([schema.wires, wires]):
            raise ValueError(
                f"Found wires either not in schema, or not in tensors: {unexpected_wires}"
            )

        for wire, tensor in tensors.items():
            dtype: str = qp.math.get_dtype_name(tensor)
            expected_dtype = schema.get_type(wire).__qualname__

            if not dtype.startswith(expected_dtype):
                raise ValueError(
                    f"Expected type {expected_dtype} for wire {wire}. Got dtype {dtype} instead"
                )

        # Check all shapes are the same
        shapes = {qp.math.shape(t) for t in tensors.values()}
        if len(shapes) > 1:
            raise MeasurementShapeError(
                f"All tensors must have the same shape, got shapes {shapes}"
            )

    def __getitem__(self, key) -> TensorLike:
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


# todo: fill this out with any initialization etc. this should be a serializable format
class CountsResult(BaseModel):
    r"""Class for holding histogram results of CV-DV programs"""

    counts: dict[int | float | tuple[int | float | complex, ...], int]
    """Histogram of basis states or eigenvalues and their frequency"""

    wire_order: Wires | None = Field(None)
    """The order of the wires in each basis state"""

    basis_schema: sa.BasisSchema | None = Field(None)
    """Schema determining the basis each wire is measured in"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def is_eigenvals(self):
        """Whether the entries of the histogram are (scalar) eigenvalues of an observable"""
        return self.wire_order is None

    @property
    def is_basis_states(self):
        """Whether the entries of the histogram are computational basis states"""
        return self.wire_order is not None

    @property
    def shots(self):
        """Returns the total number of shots contained in this result"""
        return sum(self.counts.values())

    @model_validator(mode="after")
    def check_optional_fields(self) -> "CountsResult":
        if (self.wire_order is None) != (self.basis_schema is None):
            raise ValueError(
                "Both wire_order and basis_schema must be provided, or neither provided"
            )

        return self

    @model_validator(mode="after")
    def check_format(self) -> "CountsResult":
        # Pydantic can coerce things into the correct types above, but it can't tell if someone is mixing
        # basis states and eigenvalues in the same histogram.

        # If wires are provided, then we should have only basis states
        if self.wire_order is not None:
            for basis_state in self.counts:
                if not isinstance(basis_state, tuple):
                    raise ValueError(
                        "Since wire_order is given, entries must be basis state tuples"
                    )

                if len(basis_state) != len(self.wire_order):
                    raise ValueError("Mismatch between basis state and wire count")

        # Should have only scalar eigenvalues
        else:
            for eigval in self.counts:
                if not isinstance(
                    eigval, (int, float)
                ):  # not complex bc observables are hermitian
                    raise ValueError("Expected scalar type for eigenvalue")

        return self


class SampleMeasurement(MeasurementProcess):
    r"""Interface for all finite-sampling measurements

    Any subclass should override ``process_samples`` if it can compute its measurement result from the samples,
    and it should override ``process_counts`` if it can compute using an aggregated histogram.

    .. seealso::

        :py:class:`~pennylane.measurements.measurement.SampleMeasurement`
    """

    _shortname = "sample"

    def __init__(
        self, obs=None, schema: sa.BasisSchema | None = None, eigvals=None, id=None
    ):
        """
        Args:
            obs: The optional observable to sample from. If provided, the samples will be a set of
                eigenvalue samples.

            schema: The optional schema describing the bases each wire is measured in. If it is provided,
                the samples will be computational basis states matching the format in the schema. If an
                observable is provided, ``schema`` must be ``None``, where it will be inferred from the
                observable.

            eigvals: An optional array of the eigenvalues of an observable. If provided, the samples will be
                eigenvalues from the array. Note that this doesn't make sense for position/coherent basis measurements
                since those do not have a finite number of eigenstates.

            id: An optional identifier to label the measurement operation.
        """
        if (obs is None) == (schema is None):
            raise ValueError(
                "Can only pass observable or schema because schemas are inferred from an observable"
            )

        if schema is None and obs is not None:
            schema = sa.infer_schema_from_observable(obs)
            wires = None
        elif schema is not None and obs is None:
            wires = schema.wires

        assert schema is not None

        self.schema = schema
        super().__init__(obs=obs, wires=wires, eigvals=eigvals, id=id)  # type: ignore

    @abstractmethod
    def process_samples(
        self,
        samples: SampleResult,
        wire_order: Wires,
        shot_range: tuple[int, ...] | None = None,
        bin_size: int | None = None,
    ):
        r"""Calculate the measurement given the samples

        Args:
            samples: A :py:class:`.SampleResult` containing the samples measured on each wire

            wire_order: The order of the wires in the circuit

            shot_range: A 2-tuple specifying the range of samples to use. If not specified, all samples are used

            bin_size: Divides the shot range into bins of ``bin_size`` and then computes the result over each
                bin. If not specified, all samples are grouped into a single bin.
        """

    @abstractmethod
    def process_counts(self, counts: CountsResult):
        """Calculate the measurement from a counts histogram

        Args:
            counts: A :py:class:`.CountsResult` containing the histogram of either basis states or
                eigenvalues.
        """


# -----------------------------------
#         Statevector Methods
# -----------------------------------


class Truncation(ABC):
    """An interface for specifying truncation strategies on statevectors"""

    @abstractmethod
    def dim(self, wire: Hashable) -> int:
        r"""Returns the hilbert dimension of a wire"""

    def shape(self, wire_order: Wires) -> tuple[int, ...]:
        r"""Returns the system shape for each wire in the order provided"""
        return tuple(self.dim(wire) for wire in wire_order)

    def reshape(self, state: Sequence[complex], wire_order: Wires) -> TensorLike:
        """Reshapes a possibly-batched statevector to match the system shape of this truncation

        Args:
            state: A flattened statevector of shape ``(..., d)``, which has optional batch dimensions.

            wire_order: The order of the wires in the statevector

        Returns:
            The statevector with shape ``(..., *shape)``, where ``shape`` is the result of :py:meth:`.Truncation.shape`. Each
            wire will have its own dimension.
        """
        target_shape = self.shape(wire_order)

        state = qp.math.array(state)
        orig_shape: tuple[int, ...] = qp.math.shape(state)
        has_batch_dim = len(orig_shape) > 1

        if has_batch_dim:
            target_shape = tuple([*orig_shape[:-1], target_shape])

        return qp.math.reshape(state, target_shape)


class FockTruncation(Truncation, BaseModel):
    r"""Truncation in Fock space up to a desired photon count

    For each wire, a size should be provided indicating the dimension of that subsystem. If no size is provided
    for a wire, it is defaulted to 2 (a qubit).

    Note that we allow continuous-variable bases in the schema even though we have a hard system-size
    cutoff in Fock space. This is because someone might want to simulate position measurements while truncating
    the maximum energy of a qumode.
    """

    basis_schema: sa.BasisSchema
    """Schema holding the basis for each wire"""

    dim_sizes: dict[Hashable, int]
    """Mapping from wires to their truncated system dimension"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def dim(self, wire):
        return self.dim_sizes.get(wire, 2)

    @classmethod
    def all_fock_space(cls, wires: Sequence[Hashable], dim_sizes: dict[Hashable, int]):
        wires = Wires.all_wires(wires)
        schema = sa.BasisSchema({w: sa.ComputationalBasis.Discrete for w in wires})
        return cls(basis_schema=schema, dim_sizes=dim_sizes)

    def __eq__(self, other):
        if not isinstance(other, FockTruncation):
            return False

        return (
            self.basis_schema == other.basis_schema
            and self.dim_sizes == other.dim_sizes
        )


class StateResult(BaseModel):
    statevector: Sequence[complex]

    truncation: Truncation

    wire_order: Wires

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StateMeasurement(MeasurementProcess):
    r"""Interface for all measurements that operate directly on (classically-simulated) statevectors"""

    _shortname = "state"

    @abstractmethod
    def process_state(
        self,
        state: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        r"""Calculate the measurement result using the state

        Args:
            state: The statevector flattened with an optional batch dimension, of shape
                ``(..., d)``, where ``d`` is the product of the dimensions of each wire.

            wire_order: The order of the wires in the statevector

            wire_dims: A mapping from each wire to its associated dimension

            eigvals: Optional eigenvalue vector of shape ``(d,)`` built by the device.
        """

    @abstractmethod
    def process_density_matrix(
        self,
        dm: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        r"""Calculate the measurement result using the density matrix

        Args:
            dm: The density matrix with an optional batch dimension, of shape
                ``(..., d, d)``, where ``d`` is the product of the dimensions of each wire.

            wire_order: The order of the wires in the statevector

            wire_dims: A mapping from each wire to its associated dimension

            eigvals: Optional eigenvalue vector of shape ``(d,)`` built by the device.
        """


class ShapeRequiresWireDims(MeasurementProcess):
    wire_dims: Mapping[Hashable, int] | None = None

    def copy_with_wire_dims(self, wire_dims: Mapping[Hashable, int]) -> Self:
        """Returns a copy of this measurement with the provided wire dimensions added"""
        if self.wire_dims is not None:
            raise ValueError("This measurement already has wire dimensions")

        new = copy.copy(self)
        new.wire_dims = wire_dims
        return new

    def map_wires(self, wire_map: dict[Hashable, Hashable]) -> Self:
        new = super().map_wires(wire_map)
        if self.wire_dims is not None:
            new_wire_dims = {
                wire_map.get(wire, wire): dim for wire, dim in self.wire_dims.items()
            }
            new.wire_dims = new_wire_dims
        return new
