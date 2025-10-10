# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from collections.abc import Sequence
from functools import reduce

import pennylane as qml
from pennylane.measurements import MeasurementValue
from pennylane.measurements import SampleMP as OldSampleMP
from pennylane.operation import Operator
from pennylane.ops import Pow, Prod, SProd
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from ..ops import attributes
from ..ops.mixins import Spectral
from ..sa import BasisSchema, ComputationalBasis
from .base import CountsResult, SampleMeasurement, SampleResult


def sample(
    op: Operator | MeasurementValue | Sequence[MeasurementValue] | None = None,
    schema: BasisSchema | None = None,
):
    r"""Samples the supplied observable or the wires in a provided schema

    If an observable is given, this method samples the eigenspectrum (!) of the observable, returning
    one eigenvalue per shot. The observable must have a recipe for diagonalizing it in some set of
    computational bases.

    When sampling wires without an observable, unlike in the DV (qubit) case, there are multiple possible
    computational bases to use (see :py:class:`~hybridlane.measurements.measurement.ComputationalBasis`).
    Therefore, one must provide a :py:class`~hybridlane.measurements.measurement.BasisSchema` specifying what
    basis to measure each wire in. Note that it'll likely be easier to perform homodyne (:math:`\hat{x}`) or Fock
    (:math:`\hat{n}`) measurements by supplying the corresponding observable rather than manually constructing
    the appropriate schema.
    """

    if isinstance(op, MeasurementValue):
        raise NotImplementedError("Mid-circuit measurement is currently not supported")

    return SampleMP(obs=op, schema=schema)


class SampleMP(SampleMeasurement):
    _shortname = "sample"

    @property
    def numeric_type(self):
        return float

    def shape(self, shots: int | None = None, num_device_wires: int = 0) -> tuple:
        return (shots,)

    def process_samples(
        self,
        samples: SampleResult,
        wire_order: Wires,
        shot_range: tuple[int, ...] | None = None,
        bin_size: int | None = None,
    ) -> TensorLike | SampleResult | list[TensorLike | SampleResult]:
        if shot_range or bin_size:
            # todo: handle possible shot_range or bin_size
            raise NotImplementedError("Shot range and bin size are not yet supported")

        # Pass through basis states
        if self.obs is None:
            return samples

        # If we make it here, our observable has a CV (spectral) component
        if not samples.is_basis_states:
            raise ValueError("Already provided eigenvalues")

        eigvals = self._sample_observable(self.obs, samples)
        return eigvals

    def process_counts(self, counts: CountsResult) -> CountsResult:
        # todo:
        raise NotImplementedError("Unclear how to bin potentially real eigenvalues")

    def _has_spectral_part(self, obs: Operator):
        if obs.pauli_rep is not None:
            return False

        if isinstance(obs, Spectral):
            return True

        if isinstance(obs, Prod):
            if any(isinstance(o, Spectral) for o in obs.operands):
                return True

        return False

    def _sample_observable(
        self, obs: Operator | Prod | SProd, result: SampleResult
    ) -> TensorLike:
        r"""Samples the eigenvalues of an observable

        Args:
            obs: The observable

            result: The sample results returned from the device (basis states)

        Returns:
            a tensor of shape ``(*batch_dim, shots)`` corresponding to the eigenvalue of each shot

        Raises:
            :py:class:`ValueError` if the observable has different natural bases on different wires, or if
                the result was measured in a basis besides the natural basis (and therefore the observable
                is not diagonal)
        """
        # todo: should these be in here, or should these be tape transform strategies with associated
        # postprocessing_fn? Each can be measured in a single circuit.

        # For a scalar observable c * O, eigenvalues are just scaled too
        if isinstance(obs, SProd):
            return obs.scalar * self._sample_observable(obs.base, result)

        # For an observable O^d, its eigenvalues are also raised to the d-th power
        elif isinstance(obs, Pow):
            return self._sample_observable(obs.base, result) ** obs.scalar

        # For an observable that is a tensor product O = PQ, we calculate the eigenvalues
        # for P and Q separately, then element-wise multiply them to obtain the eigenvalues
        # of O.
        elif isinstance(obs, Prod):
            tensors = [self._sample_observable(o, result) for o in obs.operands]

            # Most tensor libraries define * as elementwise multiplication. If this somehow is interpreted
            # as matmul or dot, that might become an issue
            return reduce(lambda x, y: x * y, tensors)

        # For CV operators where we defined the infinite-spectrum functions in position or fock basis,
        # use those to convert the basis states to eigenvalues
        elif (
            obs in attributes.diagonal_in_fock_basis
            or obs in attributes.diagonal_in_position_basis
        ):
            wires = obs.wires
            obs_bases = {self.schema.get_basis(w) for w in wires}
            result_bases = {result.schema.get_basis(w) for w in wires}
            basis = next(iter(result_bases))

            # We don't know how to handle mixed bases. An operator like n_1 p_0 is valid, but because
            # n_1 is in fock basis and p_0 is in position basis, they should be separated into a qml.Prod
            # where we'd calculate the eigenvalues like above. This process should be handled by circuit transforms,
            # so if we reach this point, we just error.
            if len(obs_bases) > 1:
                raise ValueError(
                    "Observable is measured across multiple bases. It should be decomposed into"
                    " a tensor product (`qml.Prod`) of single-basis observables"
                )

            # Our observable is not diagonal in the measured basis. There should have been a tape transform that
            # diagonalized it prior to us reaching this point
            if result_bases != obs_bases:
                raise ValueError(
                    f"This observable is not diagonal in the basis {result_bases} measured"
                )

            ordered_tensors = [result[w] for w in wires]

            match basis:
                case ComputationalBasis.Discrete:
                    eigvals = obs.fock_spectrum(*ordered_tensors)
                case ComputationalBasis.Position:
                    eigvals = obs.position_spectrum(*ordered_tensors)
                case _:
                    raise ValueError(
                        f"Unknown how to calculate spectrum for basis {basis}"
                    )

            return eigvals

        # All regular observables fall here, we should just dispatch to pennylane and call it a day.
        # Technically, qml.Hamiltonian/Sum also fall here, but if it errors, that's fine since they should
        # be decomposed using circuit transforms
        else:
            # Concatenate all of the (qubit) basis states together into a single tensor, so we go from
            # |wires| tensors of shape (*batch_dim, shots) to 1 tensor of shape (*batch_dim, shots, wires)
            ordered_tensors = [result[w] for w in obs.wires]
            batch_dim: tuple[int, ...] = result.shape
            ordered_tensors = [
                qml.math.reshape(t, (*batch_dim, 1)) for t in ordered_tensors
            ]
            new_samples: TensorLike = qml.math.concatenate(
                ordered_tensors, axis=-1
            )  # n x (..., 1) -> (..., n)
            with qml.QueuingManager.stop_recording():
                eigvals = OldSampleMP(
                    obs,
                    wires=None,
                    eigvals=self._eigvals,
                ).process_samples(
                    new_samples,
                    wire_order=obs.wires,
                )

            return eigvals
