# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Utilities related to manipulating tensors"""

import autoray as ar


def _numpy_unstack(arr, axis=0):
    import numpy as np

    return np.unstack(arr, axis=axis)


def _jax_unstack(arr, axis=0):
    import jax.numpy as jnp

    return jnp.unstack(arr, axis=axis)


def _torch_unstack(arr, axis=0):
    import torch  # ty:ignore[unresolved-import]

    return torch.unbind(arr, dim=axis)


ar.register_function("numpy", "unstack", _numpy_unstack)
ar.register_function("jax", "unstack", _jax_unstack)
ar.register_function("torch", "unstack", _torch_unstack)
