# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Quantum information related math functions"""

from collections.abc import Sequence
from string import ascii_lowercase as alphabet

from pennylane import math
from pennylane.typing import TensorLike


def reduce_statevector(
    state: TensorLike,
    indices: Sequence[int],
    dims: Sequence[int],
    c_dtype: str = "complex128",
) -> TensorLike:
    r"""Reduces a statevector to a density matrix by tracing out the subsystems

    Args:
        state: The statevector to be reduced. Shape (..., d) where d is the product of the
            dimensions in ``dims``.

        indices: The indices of the subsystems to keep

        dims: The dimensions of the subsystems. The length of this sequence should match the
            number of subsystems in the statevector, and the product of the dimensions should
            match the last dimension of ``state``.

        c_dtype: The complex dtype to use for the output density matrix.

    Returns:
        The reduced density matrix. Shape ``(..., d_out, d_out)`` where d_out is the product
        of the dimensions of the subsystems in ``indices``.

    **Examples**

    >>> state = np.array([1, 0, 1, 0]) / np.sqrt(2)
    >>> reduce_statevector(state, indices=[0], dims=[2, 2])
    array([[0.5+0.j, 0.5+0.j],
           [0.5+0.j, 0.5+0.j]])

    >>> reduce_statevector(state, indices=[1], dims=[2, 2])
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])

    >>> state = hl.math.kron([1, 0], [0, 1, 0])
    >>> reduce_statevector(state, indices=[0], dims=[2, 3])
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]])

    >>> state = jnp.array(state)
    >>> reduce_statevector(state, indices=[1], dims=[2, 3])
    Array([[0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex128)

    .. seealso:: :func:`~hybridlane.math.reduce_dm`
    """
    n_input = len(dims)
    d = int(math.prod(dims))

    # Reshape the statevector from (..., d) to (..., d1, d2, ..., dn)
    inner_shape = tuple(dims[i] for i in range(n_input))
    batch_size = math.get_batch_size(state, (d,), d)
    shape = (batch_size, *inner_shape) if batch_size else inner_shape
    state = math.reshape(state, shape)
    state = math.cast(state, c_dtype)

    # Construct matching indices that look like ...abcd
    indices0 = alphabet[:n_input]
    indices1 = alphabet[:n_input]

    # Now for each index we'd like to keep, assign a new unique letter
    #
    # e.g. if n_input=4 and indices=[0, 2], we should end up with
    #
    #   ...abcd,...ebfd->...aecf
    for i in indices:
        indices1 = indices1.replace(indices1[i], alphabet[n_input + i])

    deleted = set(indices0) & set(indices1)
    out_indices = "".join([c for c in indices0 + indices1 if c not in deleted])

    einsum_str = f"...{indices0},...{indices1}->...{out_indices}"
    rho = math.einsum(einsum_str, state, math.conj(state))

    # Finally reshape our density matrix to shape (..., d_out, d_out)
    d_out = int(math.prod([dims[i] for i in indices]))
    shape = (batch_size, d_out, d_out) if batch_size else (d_out, d_out)
    return math.reshape(rho, shape)


def reduce_dm(
    rho: TensorLike,
    indices: Sequence[int],
    dims: Sequence[int],
    c_dtype: str = "complex128",
) -> TensorLike:
    r"""Reduces the dimension of a density matrix by tracing out the subsystems

    Args:
        rho: The density matrix to be reduced. Shape ``(..., d, d)`` where d is the product
            of the dimensions in ``dims``.

        indices: The indices of the subsystems to keep

        dims: The dimensions of the subsystems. The length of this sequence should match the
            number of subsystems in ``rho``, and the product of the dimensions should
            match the last dimension of ``rho``.

        c_dtype: The complex dtype to use for the output density matrix.

    Returns:
        The reduced density matrix. Shape ``(..., d_out, d_out)`` where d_out is the product
        of the dimensions of the subsystems in ``indices``.

    **Examples**

    >>> rho = [[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]
    >>> reduce_dm(rho, indices=[0], dims=[2, 2])
    array([[0.5+0.j, 0.5+0.j],
           [0.5+0.j, 0.5+0.j]])

    >>> rho = jnp.array(rho)
    >>> reduce_dm(rho, indices=[1], dims=[2, 2])
    Array([[1.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j]], dtype=complex128)

    .. seealso:: :func:`~hybridlane.math.reduce_statevector`
    """
    n_input = len(dims)
    d = int(math.prod(dims))

    # Reshape from (..., d, d) to (..., d1, d2, ..., dn, d1', d2', ..., dn')
    inner_shape = tuple(dims) * 2
    batch_size = math.get_batch_size(rho, (d, d), d**2)
    shape = (batch_size, *inner_shape) if batch_size else inner_shape
    rho = math.reshape(rho, shape)
    rho = math.cast(rho, c_dtype)

    # Construct matching indices that look like ...abcd
    indices0 = alphabet[:n_input]
    indices1 = alphabet[:n_input]

    # Now for each index we'd like to keep, assign a new unique letter
    #
    # e.g. if n_input=4 and indices=[0, 2], we should end up with
    #
    #   ...abcd,...ebfd->...aecf
    for i in indices:
        indices1 = indices1.replace(indices1[i], alphabet[n_input + i])

    deleted = set(indices0) & set(indices1)
    out_indices = "".join([c for c in indices0 + indices1 if c not in deleted])

    einsum_str = f"...{indices0}{indices1}->...{out_indices}"
    rho_reduced = math.einsum(einsum_str, rho)

    # Reshape to (..., d_out, d_out)
    d_out = int(math.prod([dims[i] for i in indices]))
    shape = (batch_size, d_out, d_out) if batch_size else (d_out, d_out)
    return math.reshape(rho_reduced, shape)
