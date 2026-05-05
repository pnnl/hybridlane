# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""A wrapper around :mod:`pennylane.math` providing tensor-library agnostic functions."""

import autoray as ar
from pennylane import math as pl_math

from .matrix_manipulation import expand_matrix, expand_vector

dag = ar.dag


def __getattr__(name):
    # Fallback to pennylane math library so that users can just use our version seamlessly
    return getattr(pl_math, name)


def __dir__():
    # Include the functions from pennylane math library in the dir() output
    our_stuff = set(list(globals().keys())) - {"pl_math", "ar"}
    return list(our_stuff | set(pl_math.__dir__()))


__all__ = ["expand_matrix", "expand_vector", "dag"] + pl_math.__all__
