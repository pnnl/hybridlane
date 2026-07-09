# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import autoray as ar


def numpy_unstack(arr, axis=0):
    import numpy as np

    return np.unstack(arr, axis=axis)


def jax_unstack(arr, axis=0):
    import jax.numpy as jnp

    return jnp.unstack(arr, axis=axis)


def torch_unstack(arr, axis=0):
    import torch  # ty:ignore[unresolved-import]

    return torch.unbind(arr, dim=axis)


ar.register_function("numpy", "unstack", numpy_unstack)
ar.register_function("jax", "unstack", jax_unstack)
ar.register_function("torch", "unstack", torch_unstack)
