# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Hashable

import pennylane as qml
from pennylane.measurements import (
    MeasurementProcess,
    MeasurementShapeError,
)
from pennylane.typing import TensorLike
from pennylane.wires import Wires
from pydantic import BaseModel, ConfigDict, Field, model_validator

import hybridlane.sa as sa  # fixes a circular import

# -----------------------------------
#           Sampling Methods
# -----------------------------------


# todo: maybe turn this into a basemodel too, with deserialization from list -> numpy or optional framework?
# this should also be serializable
class SampleResult(Mapping):
    r"""Container for the results of a sample-based measurement

    This class maps wires to arrays of the corresponding type required to contain the result (see
    :py:attr:`.ComputationalBasis.return_type`).
    """

    # Design: In pennylane, basis states and eigvals are grouped together into a single samples tensor because
    # one can deduce each case by the tensor shape. Here we split it up explicitly because we split up the
    # basis states by wires to have tensors of different types

    def __init__(
        self,
        basis_states: dict[Hashable, TensorLike] | None = None,
        eigvals: TensorLike | None = None,
        schema: sa.BasisSchema | None = None,
    ):
        r"""
        Args:
            samples: A mapping from each wire to the samples drawn from it. Each sample tensor must have the same
                shape ``(..., shots)``, consisting of optional broadcast dimensions and then a common number of shots.

            eigvals: A tensor of shape ``(..., shots)`` containing the eigenvalue samples of the observable.

            schema: The schema of the basis measured for each wire. If not provided, it is inferred from the dtype of each tensor

        Raises:
            :py:class:`~pennylane.measurements.MeasurementShapeError`: if the tensors do not all have the same shape

            :py:class:`ValueError`: if the sample tensors do not match a provided schema
        """

        if (basis_states is None) == (eigvals is None):
            raise ValueError(
                "Must provide either computational basis states or eigenvalues, but not both"
            )

        self._basis_states = basis_states
        self._eigvals = eigvals

        if basis_states is not None:
            shapes: set[tuple[int, ...]] = {
                qml.math.shape(t) for t in basis_states.values()
            }
            if len(shapes) > 1:
                raise MeasurementShapeError(
                    f"All sample tensors must have the same shape. Got {shapes}"
                )
            self._shape = list(shapes)[0]

            if schema is None:
                self._schema = sa.infer_schema_from_tensors(basis_states)
            else:
                self.validate_sample_tensors(schema, basis_states)
                self._schema = schema

        elif eigvals is not None:
            self._shape = qml.math.shape(eigvals)

            # No schema necessary if we just have a list of eigenvalues

        self._ndim = len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def has_batch_dim(self):
        return self._ndim > 1

    @property
    def batch_shape(self):
        return self._shape[:-1]

    @property
    def shots(self):
        return self._shape[-1]

    @property
    def schema(self):
        return self._schema

    @property
    def eigvals(self):
        return self._eigvals

    @property
    def is_eigvals(self):
        return self._eigvals is not None

    @property
    def basis_states(self):
        return self._basis_states

    @property
    def is_basis_states(self):
        return self._basis_states is not None

    def concatenate(self, other: "SampleResult") -> "SampleResult":
        # Automatically fails if one is eigenvalues and the other is basis states
        if self.schema != other.schema:
            raise ValueError("Schemas of each result must match")

        if self.batch_shape != other.batch_shape:
            raise ValueError("Results must have the same outer dimension")

        new_tensors = eigvals = None

        if self.is_basis_states:
            new_tensors = {}
            for w in self._basis_states.keys():
                new_tensors[w] = qml.math.concatenate(
                    [self._basis_states[w], other._basis_states[w]], axis=-1
                )
        else:
            eigvals = qml.math.concatenate([self._eigvals, other._eigvals], axis=-1)

        return SampleResult(
            basis_states=new_tensors, eigvals=eigvals, schema=self.schema
        )

    def slice(self, indices: slice):
        new_tensors = eigvals = None

        if self.is_basis_states:
            new_tensors = {}
            for w in self.basis_states.keys():
                new_tensors[w] = self.basis_states[w][..., indices]
        else:
            eigvals = self.eigvals[..., indices]

        return SampleResult(
            basis_states=new_tensors, eigvals=eigvals, schema=self.schema
        )

    @staticmethod
    def validate_sample_tensors(
        schema: sa.BasisSchema, tensors: dict[Hashable, TensorLike]
    ):
        r"""Checks that provided tensors match the schema

        Args:
            tensors: The set of wire-tensor pairs to check

        Raises:
            :py:class:`ValueError`: if any of the tensors don't have the expected data type, or if a wire
                is present in the schema but not in the tensors, or if a wire is present in the tensors but
                not the schema
        """
        wires = Wires(tensors.keys())
        if unexpected_wires := Wires.unique_wires([schema.wires, wires]):
            raise ValueError(
                f"Found wires either not in schema, or not in tensors: {unexpected_wires}"
            )

        for wire, tensor in tensors.items():
            dtype: str = qml.math.get_dtype_name(tensor)
            expected_dtype = schema.get_type(wire).__qualname__

            if not dtype.startswith(expected_dtype):
                raise ValueError(
                    f"Expected type {expected_dtype} for wire {wire}. Got dtype {dtype} instead"
                )

    # These methods are only compatible if it's a basis state sample result, as they provide methods
    # for iterating over the wire-tensor pairs
    def __getitem__(self, key):
        return self._basis_states[key]

    def __iter__(self):
        return iter(self._basis_states)

    def __len__(self):
        return len(self._basis_states)


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

        state = qml.math.array(state)
        orig_shape: tuple[int, ...] = qml.math.shape(state)
        has_batch_dim = len(orig_shape) > 1

        if has_batch_dim:
            target_shape = tuple([*orig_shape[:-1], target_shape])

        return qml.math.reshape(state, target_shape)


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
        schema = sa.BasisSchema(
            {Wires(w): sa.ComputationalBasis.Discrete for w in wires}
        )
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

    # My notes: Redo state measurement to hold statevectors with arbitrary shape `(..., d1, ..., dn)`
    # if necessary. Truncation can probably be passed into our version because state measurements are explicitly
    # classical simulation. For fock state wavefunctions, this is pretty obvious, but I'm not sure how we say that
    # certain qumodes' states are represented in phase space. Gaussian wavefunctions are symplectic and can be
    # stored with the appropriate group theory representation like in StrawberryFields. For more general wavefunctions,
    # this is likely where the x/p truncation Yuan was talking about will live. Maybe mark it as `NotImplementedError` for now.

    @abstractmethod
    def process_state(self, state: StateResult):
        r"""Calculate the measurement result using the state

        Args:
            state: The statevector with associated truncation and wire order
        """
