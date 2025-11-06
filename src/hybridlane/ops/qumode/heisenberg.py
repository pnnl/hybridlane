# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

r"""A utility module to simplify working with the symplectic representations of gaussian gates."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag


def rotation(theta: float, include_constant: bool = True) -> NDArray[np.floating]:
    r"""Creates a phase-space symplectic rotation matrix

    By default, the returned matix acts on :math:`\begin{pmatrix}1 & \hat{x} & \hat{p}\end{pmatrix}^T`.

    Args:
        theta: the rotation angle

        include_constant: If set to False, the returned rotation matrix will only act on
            :math:`\begin{pmatrix}x & p\end{pmatrix}^T`

    Returns:
        The symplectic matrix describing a rotation in phase space
    """
    r = np.array(
        [
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ]
    )

    if include_constant:
        const = np.atleast_2d(1)
        r = block_diag(const, r)

    return r


def _symplectic_conversion_matrix(
    n_modes: int,
    include_constant: bool = True,
    hbar: float = 1,
    units: Literal["standard", "wigner"] = "standard",
    inv: bool = False,
) -> NDArray[np.complexfloating]:
    r"""Returns a transformation matrix from phase space to Fock space"""

    # Transformation on individual modes
    #   a = (x + ip) / 2l
    #   ad = (x - ip) / 2l
    unit_map = {"standard": np.sqrt(hbar / 2), "wigner": np.sqrt(hbar) / 2}
    R = np.array([[1, 1j], [1, -1j]]) / (2 * unit_map[units])

    # Allow inv keyword because it's cheaper to invert the 2x2 matrix here than to invert
    # the (2n+1) x (2n+1) matrix later
    if inv:
        R = np.linalg.inv(R)

    args = (R,) * n_modes
    if include_constant:
        args = (np.atleast_2d(1),) + args

    M = block_diag(*args)
    return M


def to_fock_space(
    S: NDArray[np.floating | np.complexfloating],
    hbar: float = 1,
    units: Literal["standard", "wigner"] = "standard",
) -> NDArray[np.floating | np.complexfloating]:
    r"""Transforms a symplectic matrix from phase space to fock space

    Args:
        S: A symplectic transformation matrix acting on a phase space vector :math:`(1~x_1~p_1\dots x_n~p_n)^T`

        hbar: Value for the hbar constant

        units: Units to use for the conversion, defaults to standard units :math:`\lambda_x = \lambda_p = \sqrt{\hbar/2}`

    Returns:
        An equivalent symplectic matrix acting on the creation/annihilation operators :math:`(1~a_1~\ad_1\dots a_n~\ad_n)^T`

    Raises:
        ValueError if the matrix S is not square
    """
    S = np.atleast_2d(S)

    if not S.shape[0] == S.shape[1]:
        raise ValueError(f"Expected a square symplectic matrix, got shape {S.shape}")

    n_modes, include_constant = S.shape[0] // 2, bool(S.shape[0] % 2)
    T = _symplectic_conversion_matrix(n_modes, include_constant, hbar=hbar, units=units)
    Tinv = np.linalg.inv(T)

    return T @ S @ Tinv


def to_phase_space(
    S: NDArray[np.floating | np.complexfloating],
    hbar: float = 1,
    units: Literal["standard", "wigner"] = "standard",
) -> NDArray[np.floating | np.complexfloating]:
    r"""Transforms a symplectic matrix from Fock space to phase space

    Args:
        S: A symplectic transformation matrix acting on a Fock space vector :math:`(1~a_1~\ad_1\dots a_n~\ad_n)^T`

        hbar: Value for the hbar constant

        units: Units to use for the conversion, defaults to standard units :math:`\lambda_x = \lambda_p = \sqrt{\hbar/2}`

    Returns:
        An equivalent symplectic matrix acting on the phase space operators :math:`(1~x_1~p_1\dots x_n~p_n)^T`

    Raises:
        ValueError if the matrix S is not square
    """
    S = np.atleast_2d(S)

    if not S.shape[0] == S.shape[1]:
        raise ValueError(f"Expected a square symplectic matrix, got shape {S.shape}")

    n_modes, include_constant = S.shape[0] // 2, bool(S.shape[0] % 2)
    T = _symplectic_conversion_matrix(
        n_modes, include_constant, hbar=hbar, units=units, inv=True
    )
    Tinv = np.linalg.inv(T)

    # Symplectic matrices in phase space are real
    return (T @ S @ Tinv).real


def get_antisymmetric_matrix(
    n_modes: int, include_constant: bool = True
) -> NDArray[np.floating]:
    r"""Returns the antisymmetric matrix :math:`\Omega`

    This is likely not of much use externally, but it returns the matrix

    .. math::

        \Omega = \bigoplus_n \begin{pmatrix} 0 & 1\\ -1 & 0 \end{pmatrix}

    which is used to test if a matrix is symplectic

    Args:
        n_modes: The number of qumodes :math:`n` the matrix should be defined over

        include_constant: If true (default), this augments the matrix with a constant term,
            returning :math:`0 \oplus \Omega`

    Returns:
        The antisymmetric matrix
    """
    B = np.array([[0, 1], [-1, 0]])
    args = (B,) * n_modes
    if include_constant:
        args = (np.atleast_2d(0),) + args

    omega = block_diag(*args)
    return omega


def is_symplectic(S: NDArray[np.floating]) -> bool:
    r"""Tests if a matrix is symplectic

    This checks the validity of the equation

    .. math::

        S^T \Omega S = \Omega

    where :math:`\Omega` is the antisymmetric matrix (see :py:func:`~.get_antisymmetric_matrix`).

    Args:
        S: The matrix to test

    Returns:
        True if the equation is satisfied (up to machine precision)
    """
    S = np.atleast_2d(S)

    if not S.shape[0] == S.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {S.shape}")

    n_modes, include_constant = S.shape[0] // 2, bool(S.shape[0] % 2)
    omega = get_antisymmetric_matrix(n_modes, include_constant)

    # eq. 101 in arXiv:2407.10381
    res = S.T @ omega @ S

    # Don't compare the constant parts
    if include_constant:
        res = res[1:, 1:]
        omega = omega[1:, 1:]

    return np.allclose(res, omega)
