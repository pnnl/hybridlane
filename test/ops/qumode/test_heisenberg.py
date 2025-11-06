# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import numpy as np
import pytest

import hybridlane as hqml


class TestRotation:
    @pytest.mark.parametrize("t", np.linspace(0, 2 * np.pi, 10))
    def test_angle(self, t):
        r = hqml.heisenberg.rotation(t)
        expected = np.array(
            [
                [1, 0, 0],
                [0, np.cos(t), np.sin(t)],
                [0, -np.sin(t), np.cos(t)],
            ],
        )
        assert np.allclose(r, expected)


class TestIsSymplectic:
    def test_identity(self):
        S = np.eye(3)
        assert hqml.heisenberg.is_symplectic(S)

    def test_displacement(self):
        S = np.array(
            [
                [1, 0, 0],
                [0.5, 1, 0],
                [0.5, 0, 1],
            ]
        )

        assert hqml.heisenberg.is_symplectic(S)

    def test_rotation(self):
        S = hqml.heisenberg.rotation(0.5)
        assert hqml.heisenberg.is_symplectic(S)
