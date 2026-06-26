# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Utilities for working with symplectic matrices.

Convention
----------
Throughout this module the phase-space quadrature operators satisfy the
canonical commutation relation

.. math::

    [\hat{x}, \hat{p}] = i \quad (\hbar = 1)

which fixes the ladder-operator definitions

.. math::

    a = \frac{\hat{x} + i\hat{p}}{\sqrt{2}}, \qquad
    a^\dagger = \frac{\hat{x} - i\hat{p}}{\sqrt{2}}

Like PennyLane, symplectic matrices are represented in the xpxp ordering with an additional
constant term

.. math::

    \mathbf{r} = (1,\, \hat{x}_1,\, \hat{p}_1,\, \dots,\, \hat{x}_n,\, \hat{p}_n)^\top

For functions that convert between the phase-space basis and the Fock-space
ladder-operator basis, the resulting ordering is still interleaving:

.. math::

    \mathbf{f} = (1,\, a_1,\, a^\dagger_1,\, \dots,\, a_n,\, a^\dagger_n)^\top
"""

import math as cmath

from pennylane import math
from pennylane.typing import TensorLike

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "rotation",
    "symplectic_form",
    "is_symplectic",
    "to_fock_space",
    "to_phase_space",
]


def rotation(theta: TensorLike, include_constant: bool = True) -> TensorLike:
    r"""Symplectic rotation matrix in phase space.

    Returns the matrix :math:`R(\theta)` that describes a rotation of the
    phase-space quadratures by angle :math:`\theta`:

    .. math::

        \begin{pmatrix} \hat{x}' \\ \hat{p}' \end{pmatrix}
        = \begin{pmatrix} \cos\theta & \sin\theta \\
                         -\sin\theta & \cos\theta \end{pmatrix}
          \begin{pmatrix} \hat{x} \\ \hat{p} \end{pmatrix}

    Args:
        theta: The rotation angle

        include_constant: If ``True`` (default) the returned matrix acts on the
            extended basis :math:`(1, \hat{x}, \hat{p})^\top` and has shape
            ``(3, 3)``.  If ``False`` only the :math:`2 \times 2` quadrature
            block is returned.

    Returns:
        The symplectic rotation matrix with the same array interface as
        ``theta``.

    Examples:

    >>> rotation(np.pi / 2)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0., -1.,  0.]])

    >>> rotation(np.pi / 2, include_constant=False)
    array([[ 0.,  1.],
           [-1.,  0.]])
    """

    # see box IV.2 of liu2026hybrid
    c = math.cos(theta)
    s = math.sin(theta)
    # eq 147
    m = math.asarray(
        [
            [c, s],
            [-s, c],
        ],
        like=theta,
    )
    if include_constant:
        m = math.block_diag(
            [
                math.asarray([[1.0]], like=theta),
                m,
            ]
        )

    return m


def symplectic_form(n_modes: int, like: str | None = None) -> TensorLike:
    r"""The symplectic form :math:`\Omega` for :math:`n` modes.

    Returns the :math:`2n \times 2n` antisymmetric matrix

    .. math::

        \Omega = \bigoplus_{k=1}^{n}
                 \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}

    This matrix satisfies :math:`\Omega^\top = -\Omega`.

    Args:
        n_modes: Number of bosonic modes.

        like: Optional interface specifier for the returned array.  If ``None`` (default) the
            array interface is inferred from the context.

    Returns:
        The :math:`2n \times 2n` symplectic form matrix.

    Examples:

    >>> symplectic_form(1)
    array([[ 0.,  1.],
           [-1.,  0.]])

    >>> symplectic_form(2)
    array([[ 0.,  1.,  0.,  0.],
           [-1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.],
           [ 0.,  0., -1.,  0.]])
    """

    # eq. 49 of https://www.ams.org/notices/200705/fea-mezzadri-web.pdf
    identity = math.eye(n_modes, like=like, dtype="int32")
    e2 = math.asarray([[0, 1], [-1, 0]], like=like, dtype="int32")
    return math.cast(math.kron(identity, e2), dtype="float64")


def is_symplectic(S: TensorLike, rtol=1e-5, atol=1e-8) -> bool:
    r"""Test whether a matrix is symplectic.

    A matrix :math:`S` is symplectic if its :math:`2n \times 2n` quadrature
    block :math:`\tilde{S}` satisfies

    .. math::

        \tilde{S}^\top\, \Omega\, \tilde{S} = \Omega

    where :math:`\Omega` is the symplectic form returned by
    :func:`symplectic_form`.

    If ``S`` has an odd leading dimension (i.e. it is given in the extended
    basis with an affine constant row and column), the first row and column are
    stripped automatically before the test is applied.

    Args:
        S: A square matrix of shape ``(2n, 2n)`` or ``(2n+1, 2n+1)`` with an optional
        batch dimension.

    Returns:
        ``True`` if the symplecticity condition holds to machine precision,
        ``False`` otherwise.

    Examples:

    >>> is_symplectic(np.eye(3))
    np.True_

    >>> is_symplectic(np.array([[2, 1], [0, 0]]))
    np.False_
    """
    if math.ndim(S) < 2 or ((shape := math.shape(S)) and shape[-1] != shape[-2]):
        raise ValueError(f"Input must be a square matrix, got shape {shape}")

    n = shape[-1]
    n_modes = n // 2
    has_constant = bool(n % 2)

    batch_size = math.get_batch_size(S, (n, n), n**2)
    if batch_size is not None:
        transpose_axes = (0, 2, 1)
    else:
        transpose_axes = (1, 0)

    S = math.asarray(S)
    omega = symplectic_form(n_modes, like=math.get_interface(S))
    Sc = S[..., 1:, 1:] if has_constant else S
    residual = math.transpose(Sc, axes=transpose_axes) @ omega @ Sc - omega
    return math.all(math.isclose(residual, 0, rtol=rtol, atol=atol), axis=(-1, -2))


def to_fock_space(S: TensorLike) -> TensorLike:
    r"""Convert a symplectic matrix from the phase-space to the Fock-space basis.

    Given a symplectic matrix :math:`S` acting on the extended phase-space
    basis :math:`(1, \hat{x}_1, \hat{p}_1, \dots)`, returns the equivalent
    matrix acting on the extended Fock-space basis
    :math:`(1, a_1, a^\dagger_1, \dots)`.

    Args:
        S: Symplectic matrix of shape ``(2n+1, 2n+1)`` with an optional batch dimension.

    Returns:
        The equivalent symplectic matrix in the Fock-space basis.  The result
        is complex-valued.

    Examples:

    The symplectic representation of the displacment gate in phase space mapping
    :math:`x \mapsto x + \sqrt{2}\alpha`:

    >>> S = hl.D(0.5, 0, wires=0).heisenberg_tr((0,))
    >>> S
    array([[1.    , 0.    , 0.    ],
           [0.7071, 1.    , 0.    ],
           [0.    , 0.    , 1.    ]])

    Its corresponding representation in the mode basis maps :math:`a \mapsto a + \alpha`
    and :math:`\ad \mapsto \ad + \alpha^*`:

    .. skip: next "fails sometimes with +/- 0"

    >>> hl.math.to_fock_space(S)
    array([[1. +0.j, 0. +0.j, 0. +0.j],
           [0.5+0.j, 1. +0.j, 0. +0.j],
           [0.5+0.j, 0. +0.j, 1. +0.j]])
    """
    if math.ndim(S) < 2 or ((shape := math.shape(S)) and shape[-1] != shape[-2]):
        raise ValueError(f"Input must be a square matrix, got shape {shape}")

    n = shape[-1]
    n_modes = n // 2

    T, Tinv = _fock_conversion_matrices(n_modes, like=math.get_interface(S))
    S_c = math.cast(S, "complex128")
    return T @ S_c @ Tinv


def to_phase_space(S: TensorLike):
    r"""Convert a symplectic matrix from the Fock-space to the phase-space basis.

    Inverse of :func:`to_fock_space`.  Given a symplectic matrix :math:`S`
    acting on the extended Fock-space basis
    :math:`(1, a_1, a^\dagger_1, \dots)`, returns the equivalent real-valued
    matrix acting on the extended phase-space basis
    :math:`(1, \hat{x}_1, \hat{p}_1, \dots)`.

    Args:
        S: Symplectic matrix in the Fock-space basis of shape ``(2n+1, 2n+1)``. May have
            an optional batch dimension

    Returns:
        The equivalent real-valued symplectic matrix in the phase-space basis.

    Examples:

    >>> S = hl.D(0.5, 0, wires=0).heisenberg_tr((0,))
    >>> S
    array([[1.    , 0.    , 0.    ],
           [0.7071, 1.    , 0.    ],
           [0.    , 0.    , 1.    ]])
    >>> to_phase_space(to_fock_space(S))
    array([[1.    , 0.    , 0.    ],
           [0.7071, 1.    , 0.    ],
           [0.    , 0.    , 1.    ]])
    """
    if math.ndim(S) < 2 or ((shape := math.shape(S)) and shape[-1] != shape[-2]):
        raise ValueError(f"Input must be a square matrix, got shape {shape}")

    n = shape[-1]
    n_modes = n // 2

    T, Tinv = _fock_conversion_matrices(n_modes, like=math.get_interface(S))
    S_c = math.cast(S, "complex128")
    return math.real(Tinv @ S_c @ T)


def _fock_conversion_matrices(
    n_modes: int, like: str | None = None
) -> tuple[TensorLike, TensorLike]:
    lam = 1.0 / cmath.sqrt(2)
    T2 = math.asarray([[1 + 0j, 1j], [1 + 0j, -1j]], like=like) * lam
    Tinv2 = math.asarray([[1 + 0j, 1 + 0j], [-1j, 1j]], like=like) * lam
    blocks_T = [math.eye(1, like=like)] + [T2] * n_modes
    blocks_Tinv = [math.eye(1, like=like)] + [Tinv2] * n_modes
    T = math.block_diag(blocks_T)
    Tinv = math.block_diag(blocks_Tinv)
    return T, Tinv
