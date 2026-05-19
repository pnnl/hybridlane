# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""A wrapper around :mod:`pennylane.math` providing tensor-library agnostic functions."""

import autoray as ar
from pennylane import math as pl_math

from . import array_manipulation  # noqa: F401
from .matrix_manipulation import expand_matrix, expand_vector
from .quantum import reduce_dm, reduce_statevector

dag = ar.dag


def __getattr__(name):
    import numpy as np

    # This branch stops Sybil/pytest from erroring when looking for "pytest_plugins" or
    # "__code__"
    if name in vars(pl_math) or name in dir(np):
        return getattr(pl_math, name)
    raise AttributeError(f"module 'hybridlane.math' has no attribute {name!r}")


def __dir__():
    # Include the functions from pennylane math library in the dir() output
    our_stuff = set(list(globals().keys())) - {"pl_math", "ar", "array_manipulation"}
    return list(our_stuff | set(pl_math.__dir__()))


__all__ = [
    "expand_matrix",
    "expand_vector",
    "reduce_dm",
    "reduce_statevector",
    "dag",
] + pl_math.__all__
