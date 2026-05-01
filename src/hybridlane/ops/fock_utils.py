# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

"""Module containing utility functions for constructing the Fock representations of operators"""

from typing import Literal

from pennylane.typing import TensorLike

from .. import math


def creation_operator(n: int, like: str = None) -> TensorLike:
    """Returns the creation operator for a single mode with n levels."""
    return math.diag([math.sqrt(i) for i in range(1, n)], k=-1, like=like)


def annihilation_operator(n: int, like: str = None) -> TensorLike:
    """Returns the annihilation operator for a single mode with n levels."""
    return math.diag([math.sqrt(i) for i in range(1, n)], k=1, like=like)


def position_operator(
    n: int,
    like: str = None,
    units: Literal["standard", "wigner"] = "standard",
) -> TensorLike:
    """Returns the position operator in the Fock basis"""

    lam = math.sqrt(1 / 2) if units == "standard" else 1 / 2
    return lam * (creation_operator(n, like=like) + annihilation_operator(n, like=like))


def momentum_operator(
    n: int,
    like: str = None,
    units: Literal["standard", "wigner"] = "standard",
) -> TensorLike:
    """Returns the momentum operator in the Fock basis"""

    lam = math.sqrt(1 / 2) if units == "standard" else 1 / 2
    return (
        -1j
        * lam
        * (annihilation_operator(n, like=like) - creation_operator(n, like=like))
    )
