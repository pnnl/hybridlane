# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy as sp
from pennylane import math
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike
from scipy.sparse import coo_array, sparray
from typing_extensions import overload

if TYPE_CHECKING:
    from torch import Tensor as TorchTensor  # ty:ignore[unresolved-import]


@overload
def expand_matrix(
    mat: TensorLike,
    wires: WiresLike,
    wire_dims: Mapping[Any, int] | None = None,
    wire_order: WiresLike | None = None,
    sparse_format: str = "csr",
) -> TensorLike: ...


@overload
def expand_matrix(
    mat: sparray,
    wires: WiresLike,
    wire_dims: Mapping[Any, int] | None = None,
    wire_order: WiresLike | None = None,
    sparse_format: str = "csr",
) -> sparray: ...


def expand_matrix(
    mat: TensorLike | sparray,
    wires: WiresLike,
    wire_dims: Mapping[Any, int] | None = None,
    wire_order: WiresLike | None = None,
    sparse_format: str = "csr",
) -> TensorLike | sparray:
    r"""Equivalent of :func:`pennylane.math.expand_matrix`

    Compared to the PennyLane version, this supports variable dimensions per subsystem, passed
    in through the ``wire_dims`` argument. If not specified, it defaults to 2 for all wires, which is the behavior of PennyLane.

    Args:
        mat: The matrix to expand. Must be square and have shape ``(d, d)`` where ``d`` is
            the composite dimension of the wires in ``wires``. Can also have a batch dimension.

        wires: The wires that the matrix acts on.

        wire_dims: A mapping from wire labels to their dimensions. If not specified, all wires
            are assumed to have dimension 2.

        wire_order: The order of wires to use in the output matrix. If not specified, the
            order of wires in the output will be the same as the order of wires in the input.

        sparse_format: The sparse format to use if the input is a sparse matrix. Default is
            "csr".
    """
    # Replicate pennylane behavior by assuming qubit dimension if not specified
    if wire_dims is None:
        wire_dims = defaultdict(lambda: 2)  # type: ignore[assignment]

    wires = Wires(wires)
    wire_order = Wires(wire_order or [])
    if not wire_order or wire_order == wires:
        return mat

    shape = math.shape(mat)
    if not wires and shape == (1, 1):  # pyright: ignore[reportCallIssue]
        return complex(mat[0, 0])  # ty:ignore[invalid-argument-type, not-subscriptable]

    if wires - wire_order:
        raise ValueError("The wire_order must contain every wire in wires.")

    interface = math.get_interface(mat)
    batch_dim = shape[0] if len(shape) == 3 else None

    # First step is to find the smallest subset of `wire_order` that contains every wire in
    # `wires`. This might have extra wires, so then we need to pad it with identity and then
    # permute it to match that subset order.
    indices = [wire_order.index(w) for w in wires]
    min_idx = min(indices)
    max_idx = max(indices)
    before_wire_order = wire_order[:min_idx]
    mid_wire_order = wire_order[min_idx : max_idx + 1]
    after_wire_order = wire_order[max_idx + 1 :]
    extra_wires = (
        mid_wire_order - wires
    )  # all the wires that we'll need to put identity on

    # Now we'll pad all the extra wires with identity
    if extra_wires:
        dim = int(math.prod([wire_dims[wire] for wire in extra_wires]))
        id_mat = math.eye(dim, like=interface)
        mat = _kron_with_batch(mat, id_mat, interface, batch_dim)

    # Permute from sub_wire_order to subset, which is a subset of our target wire order.
    # After this, we'll only need to pad the beginning or end.
    curr_wire_order = (
        wires + extra_wires
    )  # subset reordered so that all extras come at the end
    if interface == "scipy":
        mat = cast(sparray, mat)
        mat = permute_sparse_matrix(
            mat, curr_wire_order, mid_wire_order, wire_dims, sparse_format
        )
    else:
        mat = cast(TensorLike, mat)
        mat = permute_dense_matrix(mat, curr_wire_order, mid_wire_order, wire_dims)

    # Now pad the before and after sections with identity too
    if before_wire_order:
        dim = int(math.prod([wire_dims[wire] for wire in before_wire_order]))
        id_mat = math.eye(dim, like=interface)
        mat = _kron_with_batch(id_mat, mat, interface, batch_dim)

    if after_wire_order:
        dim = int(math.prod([wire_dims[wire] for wire in after_wire_order]))
        id_mat = math.eye(dim, like=interface)
        mat = _kron_with_batch(mat, id_mat, interface, batch_dim)

    return mat


def _kron(
    mat1: TensorLike | sparray,
    mat2: TensorLike | sparray,
    interface: str,
) -> TensorLike | sparray:
    if interface == "torch":
        # According to pennylane, this avoids a crash
        mat1 = cast("TorchTensor", mat1).contiguous()  # pyright: ignore[reportAssignmentType]
        mat2 = cast("TorchTensor", mat2).contiguous()  # pyright: ignore[reportAssignmentType]

    if interface == "scipy":
        mat = sp.sparse.kron(mat1, mat2, format="coo")
        cast(coo_array, mat).eliminate_zeros()
        return cast(sparray, mat)

    return math.kron(mat1, mat2, like=interface)


def _kron_with_batch(
    mat1: TensorLike | sparray,
    mat2: TensorLike | sparray,
    interface: str,
    batch_dim: int | None,
) -> TensorLike | sparray:
    if batch_dim is None:
        return _kron(mat1, mat2, interface)

    matrices = [_kron(m, mat2, interface) for m in mat1]  # ty:ignore[not-iterable]
    return math.stack(matrices, like=interface)


def permute_dense_matrix(
    mat: TensorLike,
    wires: Wires,
    wire_order: Wires,
    wire_dims: Mapping[Any, int],
) -> TensorLike:
    shape = math.shape(mat)
    batch_dim = shape[0] if len(shape) == 3 else None

    # Here we identify the permutation from x -> y
    perm = wires.indices(wire_order)
    perm = tuple(perm) + tuple(i + len(wires) for i in perm)
    perm = tuple(i + bool(batch_dim) for i in perm)

    # Reshape the operator from (d, d) to (o1, ..., on, i1, ..., in) where oi == ii
    source_dims = tuple(wire_dims[wire] for wire in wires)
    expanded_shape = (batch_dim,) if batch_dim else () + source_dims + source_dims
    mat = math.reshape(mat, expanded_shape)
    mat = math.transpose(mat, perm)
    mat = math.reshape(mat, shape)
    return mat


def permute_sparse_matrix(
    mat: sparray,
    wires: Wires,
    wire_order: Wires,
    wire_dims: Mapping[Any, int],
    format: str,
) -> sparray:
    perm = tuple(wire_order.indices(wires))
    dims = tuple(wire_dims[wire] for wire in wires)
    P = build_sparse_permutation_matrix(dims, perm)
    assert P.shape[-2:] == mat.shape[-2:], (  # ty:ignore[unresolved-attribute]
        "Permutation matrix must be the same shape as the input matrix."
    )
    result = P @ mat @ P.T  # ty:ignore[unresolved-attribute, unsupported-operator]
    return result.asformat(format)


def build_sparse_permutation_matrix(
    dims: tuple[int, ...], perm: tuple[int, ...]
) -> sparray:
    """
    Build sparse permutation matrix for subsystem reordering.

    Args:
        dims: tuple of subsystem dimensions (d1, ..., dn)
        perm: permutation of subsystems (0-indexed)
              e.g., (1, 0, 2) swaps first two subsystems

    Returns:
        Sparse permutation matrix P in CSR format, shape (d, d) where d = prod(dims)
        Use as: M_permuted = P @ M @ P.T
    """
    d = int(math.prod(dims))

    # New dimensions after permutation
    new_dims = tuple(dims[i] for i in perm)

    # We'll collect the nonzero entries (row, col, data)
    rows = []
    cols = []

    # Iterate over all possible multi-indices
    # For efficiency, we iterate over flat indices and convert
    for old_idx in range(d):
        # Convert flat index to multi-index in original ordering
        multi_idx = np.unravel_index(old_idx, dims)

        # Apply permutation to get new multi-index
        new_multi_idx = tuple(multi_idx[i] for i in perm)

        # Convert back to flat index in new ordering
        new_idx = np.ravel_multi_index(new_multi_idx, new_dims)

        # P[new_idx, old_idx] = 1
        rows.append(new_idx)
        cols.append(old_idx)

    # Create sparse matrix with all 1s
    data = math.ones(d, dtype=np.int8)  # or dtype=np.float64 if preferred

    P = coo_array((data, (rows, cols)), shape=(d, d))
    return P.tocsr()


@overload
def expand_vector(
    vec: TensorLike,
    wires: WiresLike,
    wire_dims: Mapping[Any, int] | None = None,
    wire_order: WiresLike | None = None,
    sparse_format: str = "csr",
) -> TensorLike: ...


@overload
def expand_vector(
    vec: sparray,
    wires: WiresLike,
    wire_dims: Mapping[Any, int] | None = None,
    wire_order: WiresLike | None = None,
    sparse_format: str = "csr",
) -> sparray: ...


def expand_vector(
    vec: TensorLike | sparray,
    wires: WiresLike,
    wire_dims: Mapping[Any, int] | None = None,
    wire_order: WiresLike | None = None,
    sparse_format: str = "csr",
) -> TensorLike | sparray:
    r"""Equivalent of :func:`pennylane.math.expand_vector`

    Compared to the PennyLane version, this supports variable dimensions per subsystem, passed
    in through the ``wire_dims`` argument. If not specified, it defaults to 2 for all wires, which is the behavior of PennyLane.

    Args:
        vec: The vector to expand. Must have shape ``(d,)`` where ``d`` is
            the composite dimension of the wires in ``wires``. Can also have a batch dimension.

        wires: The wires that the matrix acts on.

        wire_dims: A mapping from wire labels to their dimensions. If not specified, all wires
            are assumed to have dimension 2.

        wire_order: The order of wires to use in the output vector. If not specified, the
            order of wires in the output will be the same as the order of wires in the input.

        sparse_format: The sparse format to use if the input is a sparse vector. Default is
            "csr".
    """
    # Replicate pennylane behavior by assuming qubit dimension if not specified
    if wire_dims is None:
        wire_dims = defaultdict(lambda: 2)  # type: ignore[assignment]

    wires = Wires(wires)
    wire_order = Wires(wire_order or [])
    if not wire_order or wire_order == wires:
        return vec

    shape = math.shape(vec)
    if not wires and shape == (1,):  # pyright: ignore[reportCallIssue]
        return complex(vec[0])  # ty:ignore[invalid-argument-type, not-subscriptable]

    if wires - wire_order:
        raise ValueError("The wire_order must contain every wire in wires.")

    interface = math.get_interface(vec)
    batch_dim = shape[0] if len(shape) == 2 else None

    # First step is to find the smallest subset of `wire_order` that contains every wire in
    # `wires`. This might have extra wires, so then we need to pad it with zero and then
    # permute it to match that subset order.
    indices = wire_order.indices(wires)
    min_idx = min(indices) if indices else 0
    max_idx = max(indices) if indices else len(wire_order) - 1
    before_wire_order = wire_order[:min_idx]
    mid_wire_order = wire_order[min_idx : max_idx + 1]
    after_wire_order = wire_order[max_idx + 1 :]
    extra_wires = mid_wire_order - wires

    # Now we'll pad all the extra wires with zeros
    if extra_wires:
        dim = int(math.prod([wire_dims[wire] for wire in extra_wires]))
        id_vec = math.cast_like(_identity_vector(dim, like=interface), vec)
        vec = _kron_with_batch(vec, id_vec, interface, batch_dim)

    # Permute from sub_wire_order to subset, which is a subset of our target wire order.
    # After this, we'll only need to pad the beginning or end.
    curr_wire_order = (
        wires + extra_wires
    )  # subset reordered so that all extras come at the end
    if interface == "scipy":
        vec = cast(sparray, vec)
        vec = permute_sparse_vector(
            vec, curr_wire_order, mid_wire_order, wire_dims, sparse_format
        )
    else:
        vec = cast(TensorLike, vec)
        vec = permute_dense_vector(vec, curr_wire_order, mid_wire_order, wire_dims)

    # Now pad the before and after sections with zeros too
    if before_wire_order:
        dim = int(math.prod([wire_dims[wire] for wire in before_wire_order]))
        id_vec = math.cast_like(_identity_vector(dim, like=interface), vec)
        vec = _kron_with_batch(id_vec, vec, interface, batch_dim)

    if after_wire_order:
        dim = int(math.prod([wire_dims[wire] for wire in after_wire_order]))
        id_vec = math.cast_like(_identity_vector(dim, like=interface), vec)
        vec = _kron_with_batch(vec, id_vec, interface, batch_dim)

    return vec


def _identity_vector(dim: int, like: str) -> TensorLike | sparray:
    if like == "scipy":
        vec = sp.sparse.coo_array(([1], ([0],)), shape=(dim,))
        return cast(sparray, vec)

    if like == "jax":
        vec = math.zeros(dim, like=like)
        vec = vec.at[0].set(1)
        return vec

    vec = math.zeros(dim, like=like)
    vec[0] = 1
    return vec


def permute_dense_vector(
    vec: TensorLike,
    wires: Wires,
    wire_order: Wires,
    wire_dims: Mapping[Any, int],
) -> TensorLike:
    shape = math.shape(vec)
    batch_dim = shape[0] if len(shape) == 2 else None

    # Here we identify the permutation from x -> y
    perm = wires.indices(wire_order)
    perm = tuple(i + bool(batch_dim) for i in perm)

    if batch_dim:
        perm = (0,) + perm

    source_dims = tuple(wire_dims[wire] for wire in wires)
    expanded_shape = ((batch_dim,) if batch_dim else ()) + source_dims
    vec = math.reshape(vec, expanded_shape)
    vec = math.transpose(vec, perm)
    vec = math.reshape(vec, shape)
    return vec


def permute_sparse_vector(
    vec: sparray,
    wires: Wires,
    wire_order: Wires,
    wire_dims: Mapping[Any, int],
    format: str,
) -> sparray:
    perm = tuple(wire_order.indices(wires))
    dims = tuple(wire_dims[wire] for wire in wires)
    P = build_sparse_permutation_matrix(dims, perm)
    result = P @ vec  # ty:ignore[unsupported-operator]
    return result.asformat(format)
