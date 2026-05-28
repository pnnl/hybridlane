# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
from collections.abc import Mapping
from typing import Hashable

from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike

from hybridlane.measurements.base import ShapeRequiresWireDims, StateMeasurement

from .. import math


def state() -> "StateMP":
    r"""Statevector measurement

    Analogous to Pennylane's state measurement (:py:func:`qp.state`), returning the
    statevector of the device across all the wires.

    **Example**

    .. code-block:: python
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=4)

        @qp.qnode(dev)
        def circuit():
            hl.FockState(1, [0, 1])
            qp.X(0)
            return hl.state()

    >>> circuit() # |1,1>
    array([0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.-0.j, 0.+0.j, 0.+0.j])
    """

    return StateMP()


class StateMP(StateMeasurement, ShapeRequiresWireDims):
    _shortname = "state"

    def __init__(
        self,
        wires: Wires | None = None,
        id: str | None = None,
    ):
        super().__init__(wires=wires, id=id)

    @property
    def numeric_type(self):
        return complex

    def shape(
        self, shots: int | None = None, num_device_wires: int = 0
    ) -> tuple[int, ...]:
        if self.wire_dims:
            return (int(math.prod(tuple(self.wire_dims.values()))),)

        return ()

    def process_state(
        self,
        state: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        if self.wires:
            state = math.expand_vector(
                state, wires=wire_order, wire_order=self.wires, wire_dims=wire_dims
            )

        is_f32 = math.get_dtype_name(state) in ("float32", "complex64")
        dtype = "complex64" if is_f32 else "complex128"
        return math.cast(state, dtype)

    def process_density_matrix(
        self,
        dm: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        raise ValueError("Can't return density matrix from `hl.state`")


def density_matrix(wires: WiresLike | None = None) -> "DensityMatrixMP":
    r"""Density matrix measurement

    Analogous to Pennylane's density matrix measurement (:py:func:`qp.density_matrix`),
    returning the density matrix of the device across the specified wires. If no wires are
    specified, returns the density matrix across all wires.

    **Example**

    .. code-block:: python
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=4)

        # Prepares a cat state with residual entanglement with a qubit
        @qp.qnode(dev)
        def circuit():
            qp.H(0)
            hl.CD(1.0, 0, wires=[0, 1])
            return hl.density_matrix(wires=0)

    >>> circuit()
    array([[0.5   +0.j, 0.0374+0.j],
           [0.0374+0.j, 0.5   +0.j]])
    """
    if wires is not None:
        wires = Wires(wires)

    return DensityMatrixMP(wires=wires)


class DensityMatrixMP(StateMeasurement, ShapeRequiresWireDims):
    _shortname = "density_matrix"

    def __init__(self, wires: Wires | None = None, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @property
    def numeric_type(self):
        return complex

    def shape(
        self, shots: int | None = None, num_device_wires: int = 0
    ) -> tuple[int, ...]:
        if self.wire_dims:
            if self.wires:
                dim = int(math.prod(tuple(self.wire_dims[w] for w in self.wires)))
            else:
                dim = int(math.prod(tuple(self.wire_dims.values())))
            return (dim, dim)

        return ()

    def process_state(
        self,
        state: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        is_f32 = math.get_dtype_name(state) in ("float32", "complex64")
        dtype = "complex64" if is_f32 else "complex128"

        if self.wires:
            # Expand the statevector if necessary and put all wires to be traced out
            # at the end, and put the wires we want to keep in order at the beginning, so that
            # after tracing out the resulting density matrix doesn't require more reordering
            trace_out = wire_order - self.wires
            expanded_wire_order = self.wires + trace_out
            state = math.expand_vector(
                state,
                wires=wire_order,
                wire_order=expanded_wire_order,
                wire_dims=wire_dims,
            )

            # Trace out all the undesired wires
            keep_indices = range(len(self.wires))
            dims = tuple(wire_dims[w] for w in expanded_wire_order)
            rho = math.reduce_statevector(
                state, indices=keep_indices, dims=dims, c_dtype=dtype
            )

        else:
            # If no wire order specified, do the usual outer product
            rho = math.outer(state, math.conj(state))

        return math.cast(rho, dtype)

    def process_density_matrix(
        self,
        dm: TensorLike,
        wire_order: Wires,
        wire_dims: Mapping[Hashable, int],
        eigvals: TensorLike | None = None,
    ) -> TensorLike:
        is_f32 = math.get_dtype_name(dm) in ("float32", "complex64")
        dtype = "complex64" if is_f32 else "complex128"

        if self.wires:
            # First delete all the wires that we're going to trace out
            trace_out = wire_order - self.wires
            traced_wire_order = wire_order - trace_out
            keep_indices = tuple(wire_order.index(w) for w in traced_wire_order)
            dm = math.reduce_dm(
                dm,
                indices=keep_indices,
                dims=tuple(wire_dims[w] for w in wire_order),
                c_dtype=dtype,
            )

            # Now extend the density matrix with any extra wires, with a resulting wire
            # order of traced_wire_order + extra
            extra = self.wires - wire_order
            wire_order = traced_wire_order + extra
            new_dim = int(math.prod([wire_dims[w] for w in extra]))
            id_rho = math.zeros((new_dim, new_dim), dtype=dtype)
            id_rho[0, 0] = 1
            id_rho = math.asarray(id_rho, like=dm)
            dm = math.kron(dm, id_rho)

            # Finally permute the subsystems
            dm = math.expand_matrix(
                dm, wires=wire_order, wire_order=self.wires, wire_dims=wire_dims
            )

        return math.cast(dm, dtype)
