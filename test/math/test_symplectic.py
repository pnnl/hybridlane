# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import hybridlane as hl
from hybridlane.math.symplectic import (
    is_symplectic,
    rotation,
    symplectic_form,
    to_fock_space,
    to_phase_space,
)


def _make_rotation_expected(theta):
    # eq 147 of liu2026hybrid
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])


@pytest.mark.unit
class TestRotation:
    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "theta", [0.0, math.pi / 6, math.pi / 4, math.pi / 2, math.pi, 1.234]
    )
    def test_values(self, theta, like):
        theta = hl.math.asarray(theta, like=like)
        for include_constant in [True, False]:
            M = rotation(theta, include_constant)
            assert hl.math.get_interface(M) == like
            assert is_symplectic(M)

            if not include_constant:
                M = hl.math.block_diag([np.array([[1.0]]), M])

            expected = _make_rotation_expected(theta)
            assert M == pytest.approx(expected, abs=1e-12)

    @pytest.mark.jax
    def test_jit(self):
        @jax.jit
        def f(theta):
            return rotation(theta)

        f(jnp.array(0.5))  # errors if jit fails

    @pytest.mark.jax
    def test_grad(self):
        def loss(t):
            M = rotation(t)
            return M @ jnp.array([1, 1, 0])  # pure x quadrature  # ty:ignore[unsupported-operator]

        t = jnp.array(0.5)
        grad_fn = jax.jacobian(loss)
        grad = grad_fn(t)
        assert jnp.all(jnp.isfinite(grad))

        # x' = cos(t) x, p' = -sin(t) x
        assert grad[0] == pytest.approx(0)
        assert grad[1] == pytest.approx(-math.sin(t))
        assert grad[2] == pytest.approx(-math.cos(t))


@pytest.mark.unit
class TestSymplecticForm:
    def test_antisymmetric(self):
        for n in [1, 2, 3]:
            omega = symplectic_form(n)
            assert omega.T == pytest.approx(-omega, abs=1e-12)  # ty:ignore[unresolved-attribute, unsupported-operator]

    def test_shape(self):
        for n in [1, 2, 3]:
            assert symplectic_form(n).shape == (2 * n, 2 * n)  # ty:ignore[unresolved-attribute]

    @pytest.mark.all_interfaces
    def test_interface(self, like):
        for n in [1, 2, 3]:
            omega = symplectic_form(n, like=like)
            assert hl.math.get_interface(omega) == like


@pytest.mark.unit
@pytest.mark.all_interfaces
class TestIsSymplectic:
    def test_identity_is_symplectic(self, like):
        for n in range(2, 10):  # odd dimensions include the constant
            assert is_symplectic(hl.math.eye(n, like=like))

    def test_rotation_is_symplectic(self, like):
        for theta in [0.1, 0.5, math.pi / 3]:
            theta = hl.math.asarray(theta, like=like)
            assert is_symplectic(rotation(theta))

    def test_not_symplectic(self, like):
        for n in range(2, 10):
            for _ in range(5):
                mat = np.random.rand(n, n)
                mat = hl.math.asarray(mat, like=like)
                assert not is_symplectic(
                    mat
                )  # really unlikely a random matrix is symplectic

    def test_batched(self, like):
        for n in (2, 3):
            for batch_size in (3, 5):
                mats = np.random.rand(batch_size, n, n)
                mats[-1] = rotation(
                    np.random.rand(), include_constant=n % 2 == 1
                )  # make one symplectic

                mats = hl.math.asarray(mats, like=like)
                results = is_symplectic(mats)

                assert hl.math.shape(results) == (batch_size,)
                assert not hl.math.any(results[:-1])  # ty:ignore[not-subscriptable]
                assert results[-1]  # last one is symplectic  # ty:ignore[not-subscriptable]


@pytest.mark.unit
class TestFockPhaseSpaceConversion:
    def _displacement_fock(self, a, phi, like=None):
        alpha = a * hl.math.exp(1j * phi)
        return hl.math.asarray(
            [
                [1, 0, 0],
                [alpha, 1, 0],
                [hl.math.conj(alpha), 0, 1],
            ],
            like=like,
        )

    @pytest.mark.all_interfaces
    def test_to_phase_space_displacement(self, like):
        p = hl.math.asarray([0.5, 0.3], like=like)
        S_fock = self._displacement_fock(*p, like=like)
        S_xp = to_phase_space(S_fock)
        expected = hl.Displacement._heisenberg_rep(p)

        assert hl.math.get_interface(S_xp) == like
        assert hl.math.get_dtype_name(S_xp) == "float64"  # should be real
        assert S_xp == pytest.approx(expected, abs=1e-10)

    @pytest.mark.all_interfaces
    def test_to_fock_space_displacement(self, like):
        p = hl.math.asarray([0.5, 0.3], like=like)
        S_xp = hl.Displacement._heisenberg_rep(p)
        S_fock = to_fock_space(S_xp)
        expected = self._displacement_fock(*p, like=like)

        assert hl.math.get_interface(S_fock) == like
        assert S_fock == pytest.approx(expected, abs=1e-10)

    @pytest.mark.all_interfaces
    def test_round_trip_conversion(self, like):
        for gate, params in [
            (hl.Rotation, [0.5]),
            (hl.Squeezing, [0.3, 0.4]),
            (hl.Displacement, [0.5, 0.3]),
            (hl.Beamsplitter, [0.4, 0.3]),
            (hl.TwoModeSqueezing, [0.3, 0.4]),
            (hl.TwoModeSum, [0.5]),
        ]:
            p = hl.math.asarray(params, like=like)
            M = gate._heisenberg_rep(p)
            converted = to_phase_space(to_fock_space(M))

            assert hl.math.get_interface(converted) == like
            assert converted == pytest.approx(M, abs=1e-10)

    @pytest.mark.jax
    def test_to_fock_space_jit(self):
        @jax.jit
        def f(M):
            return to_fock_space(M)

        p = jnp.array([0.5])
        M = hl.Rotation._heisenberg_rep(p)
        f(M)  # errors if jit fails

    @pytest.mark.jax
    def test_to_phase_space_jit(self):
        @jax.jit
        def f(M):
            return to_phase_space(M)

        p = jnp.array([0.3, 0.4])
        M = self._displacement_fock(*p, like="jax")
        f(M)  # errors if jit fails
