# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import itertools

import pennylane as qp
import pytest
import scipy.sparse as sp

from hybridlane import math
from hybridlane.math.matrix_manipulation import permute_dense_matrix


@pytest.mark.unit
class TestPermuteDenseMatrix:
    def test_same_wires(self):
        # Construct an operator over wire dimensions (2, 4)
        snap = math.diag(math.exp(1j * math.arange(4)))
        csnap = math.block_diag([snap, snap.conj().T])
        wires = qp.wires.Wires((0, 1))
        wire_dims = {0: 2, 1: 4}
        mat = permute_dense_matrix(csnap, wires, wires, wire_dims)

        assert math.allclose(mat, csnap)

    @pytest.mark.jax
    def test_same_wires_jax(self):
        from jax import numpy as jnp

        snap = jnp.diag(jnp.exp(1j * jnp.arange(4)))
        csnap = math.block_diag([snap, snap.conj().T])
        wires = qp.wires.Wires((0, 1))
        wire_dims = {0: 2, 1: 4}
        mat = permute_dense_matrix(csnap, wires, wires, wire_dims)

        assert math.get_interface(mat) == "jax"
        assert jnp.allclose(mat, csnap)

    def test_different_wires(self):
        # Construct an operator over wire dimensions (2, 4)
        #
        # CSnap looks like
        #   exp{diag([0, 1i, 2i, 3i]) + diag([0, -1i, -2i, -3i])}
        phase = math.exp(1j * math.arange(4))
        snap = math.diag(phase)
        csnap = math.block_diag([snap, snap.conj().T])

        # The transposed version interleaves the positive and negative phases:
        #   exp{diag([0, 0, 1i, -1i, 2i, -2i, 3i, -3i])}
        permuted_eigvals = math.zeros(8, dtype=complex)
        permuted_eigvals[::2] = phase
        permuted_eigvals[1::2] = phase.conj()
        expected_mat = math.diag(permuted_eigvals)

        wires = qp.wires.Wires((0, 1))
        wire_order = qp.wires.Wires((1, 0))
        wire_dims = {0: 2, 1: 4}
        mat = permute_dense_matrix(csnap, wires, wire_order, wire_dims)

        assert math.allclose(mat, expected_mat)

    @pytest.mark.jax
    def test_different_wires_jax(self):
        from jax import numpy as jnp

        # Construct an operator over wire dimensions (2, 4)
        #
        # CSnap looks like
        #   exp{diag([0, 1i, 2i, 3i]) + diag([0, -1i, -2i, -3i])}
        phase = jnp.exp(1j * jnp.arange(4))
        snap = jnp.diag(phase)
        csnap = math.block_diag([snap, snap.conj().T])

        # The transposed version interleaves the positive and negative phases:
        #   exp{diag([0, 0, 1i, -1i, 2i, -2i, 3i, -3i])}
        permuted_eigvals = jnp.zeros(8, dtype=complex)
        permuted_eigvals = permuted_eigvals.at[::2].set(phase)
        permuted_eigvals = permuted_eigvals.at[1::2].set(phase.conj())
        expected_mat = jnp.diag(permuted_eigvals)

        wires = qp.wires.Wires((0, 1))
        wire_order = qp.wires.Wires((1, 0))
        wire_dims = {0: 2, 1: 4}
        mat = permute_dense_matrix(csnap, wires, wire_order, wire_dims)

        assert math.get_interface(mat) == "jax"
        assert math.allclose(mat, expected_mat)


@pytest.mark.unit
class TestExpandMatrix:
    def test_expand_before(self):
        snap = math.diag(math.exp(1j * math.arange(4)))
        expected_mat = math.block_diag([snap, snap])
        wire_dims = {0: 4, 1: 2}
        mat = math.expand_matrix(snap, (0,), wire_dims=wire_dims, wire_order=(1, 0))
        assert math.allclose(mat, expected_mat)

    @pytest.mark.jax
    def test_expand_before_jax(self):
        from jax import numpy as jnp

        snap = jnp.diag(jnp.exp(1j * jnp.arange(4)))
        expected_mat = math.block_diag([snap, snap])
        wire_dims = {0: 4, 1: 2}
        mat = math.expand_matrix(snap, (0,), wire_dims=wire_dims, wire_order=(1, 0))
        assert math.get_interface(mat) == "jax"
        assert math.allclose(mat, expected_mat)

    @pytest.mark.torch
    def test_expand_before_torch(self):
        import torch

        snap = torch.diag(torch.exp(1j * torch.arange(4)))
        expected_mat = math.block_diag([snap, snap])
        wire_dims = {0: 4, 1: 2}
        mat = math.expand_matrix(snap, (0,), wire_dims=wire_dims, wire_order=(1, 0))
        assert math.get_interface(mat) == "torch"
        assert math.allclose(mat, expected_mat)

    def test_expand_after(self):
        snap = math.exp(1j * math.arange(4))
        expected_diags = math.zeros(8, dtype=complex)
        expected_diags[::2] = snap
        expected_diags[1::2] = snap
        expected_mat = math.diag(expected_diags)
        wire_dims = {0: 4, 1: 2}
        mat = math.expand_matrix(
            math.diag(snap), (0,), wire_dims=wire_dims, wire_order=(0, 1)
        )
        assert math.allclose(mat, expected_mat)

    def test_permutations(self):
        mat1 = math.eye(2)
        mat2 = 2 * math.eye(4)
        mat3 = 3 * math.eye(4)
        mats = [mat1, mat2, mat3]
        mat = math.kron(mat1, mat2)
        mat = math.kron(mat, mat3)
        wires = (0, 1, 2)
        wire_dims = {0: 2, 1: 4, 2: 4}
        for wire_order in itertools.permutations(wires):
            expanded_mat = math.expand_matrix(
                mat, wires, wire_dims=wire_dims, wire_order=wire_order
            )
            reordered_mats = [mats[wire_order.index(wire)] for wire in wires]
            expected_mat = math.kron(reordered_mats[0], reordered_mats[1])
            expected_mat = math.kron(expected_mat, reordered_mats[2])
            assert math.allclose(expanded_mat, expected_mat)

    def test_permutations_sparse(self):
        mat1 = sp.eye(2)
        mat2 = 2 * sp.eye(4)
        mat3 = 3 * sp.eye(4)
        mats = [mat1, mat2, mat3]
        mat = sp.kron(mat1, mat2)
        mat = sp.kron(mat, mat3)
        wires = (0, 1, 2)
        wire_dims = {0: 2, 1: 4, 2: 4}
        for wire_order in itertools.permutations(wires):
            expanded_mat = math.expand_matrix(
                mat, wires, wire_dims=wire_dims, wire_order=wire_order
            )
            reordered_mats = [mats[wire_order.index(wire)] for wire in wires]
            expected_mat = sp.kron(reordered_mats[0], reordered_mats[1])
            expected_mat = sp.kron(expected_mat, reordered_mats[2])
            assert math.allclose(expanded_mat, expected_mat)


@pytest.mark.unit
class TestExpandVector:
    def test_noop(self):
        vec = math.array([1, 2, 3, 4])
        wire_dims = {0: 2, 1: 2}
        expanded_vec = math.expand_vector(vec, (0, 1), wire_dims=wire_dims)
        assert math.allclose(expanded_vec, vec)

    def test_reverse(self):
        vec = math.arange(8)
        expected_vec = math.asarray([0, 4, 2, 6, 1, 5, 3, 7])
        expanded_vec = math.expand_vector(vec, (0, 1, 2), wire_order=(2, 1, 0))
        assert math.allclose(expanded_vec, expected_vec)

    @pytest.mark.jax
    def test_reverse_jax(self):
        from jax import numpy as jnp

        vec = jnp.arange(8)
        expected_vec = jnp.asarray([0, 4, 2, 6, 1, 5, 3, 7])
        expanded_vec = math.expand_vector(vec, (0, 1, 2), wire_order=(2, 1, 0))
        assert math.get_interface(expanded_vec) == "jax"
        assert math.allclose(expanded_vec, expected_vec)

    @pytest.mark.torch
    def test_reverse_torch(self):
        import torch

        vec = torch.arange(8)
        expected_vec = torch.asarray([0, 4, 2, 6, 1, 5, 3, 7])
        expanded_vec = math.expand_vector(vec, (0, 1, 2), wire_order=(2, 1, 0))
        assert math.get_interface(expanded_vec) == "torch"
        assert math.allclose(expanded_vec, expected_vec)

    def test_permuting(self):
        wire_state = {0: 0, 1: 1, 2: 2}
        state1 = math.asarray([1, 0])  # |0>
        state2 = math.asarray([0, 1, 0, 0])  # |1>
        state3 = math.asarray([0, 0, 1, 0])  # |2>
        state = math.kron(state1, state2)
        state = math.kron(state, state3)
        wires = (0, 1, 2)
        wire_dims = {0: 2, 1: 4, 2: 4}
        for wire_order in itertools.permutations(wires):
            expanded_state = math.expand_vector(
                state, wires, wire_dims=wire_dims, wire_order=wire_order
            )
            multi_idx = tuple(wire_state[wire] for wire in wire_order)
            dims = tuple(wire_dims[wire] for wire in wire_order)
            target_idx = math.ravel_multi_index(multi_idx, dims)
            nonzero_idx = math.nonzero(expanded_state)[0][0]
            assert nonzero_idx == target_idx

    def test_expand_before(self):
        null_state = math.asarray([1, 0])
        state = math.asarray([0, 0, 0, 1])
        expected_state = math.kron(null_state, state)
        wire_dims = {0: 2, 1: 4}
        expanded_state = math.expand_vector(
            state, (1,), wire_dims=wire_dims, wire_order=(0, 1)
        )
        assert math.allclose(expanded_state, expected_state)

    def test_expand_before_sparse(self):
        null_state = sp.csr_array([1, 0])
        state = sp.csr_array([0, 0, 0, 1])
        expected_state = sp.kron(null_state, state)
        wire_dims = {0: 2, 1: 4}
        expanded_state = math.expand_vector(
            state, (1,), wire_dims=wire_dims, wire_order=(0, 1)
        )
        assert math.allclose(expanded_state, expected_state)

    @pytest.mark.jax
    def test_expand_before_jax(self):
        null_state = math.asarray([1, 0], like="jax")
        state = math.asarray([0, 0, 0, 1], like="jax")
        expected_state = math.kron(null_state, state)
        wire_dims = {0: 2, 1: 4}
        expanded_state = math.expand_vector(
            state, (1,), wire_dims=wire_dims, wire_order=(0, 1)
        )
        assert math.allclose(expanded_state, expected_state)

    def test_expand_after(self):
        null_state = math.asarray([1, 0])
        state = math.asarray([0, 0, 0, 1])
        expected_state = math.kron(state, null_state)
        wire_dims = {0: 4, 1: 2}
        expanded_state = math.expand_vector(
            state, (0,), wire_dims=wire_dims, wire_order=(0, 1)
        )
        assert math.allclose(expanded_state, expected_state)
