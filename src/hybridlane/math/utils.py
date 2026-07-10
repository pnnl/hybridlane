# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Utility functions for working with tracers and concrete values."""

from collections.abc import Callable

import pennylane as qp
from pennylane.typing import TensorLike


def concrete_or_error(
    f: Callable | None, x: TensorLike, context: str = "", like: str | None = None
) -> TensorLike:
    r"""Errors if the input is a tracer that can't be concretized with ``f``

    This function behaves like ``jax.extend.core.concrete_or_error``

    Args:
        f: The optional function to use to concretize the input. If set, the return value will
            be ``f(x)``. If not set, the return value will be ``x``.

        x: The input to check for tracers

        context: The context to use in the error message if an error is raised

        like: The interface to use. If not set, the interface will be inferred from ``x``.

    Returns:
        The result of ``f(x)`` if ``f`` is not None, otherwise ``x``.

    Raises:
        ValueError: If ``x`` is a tracer that can't be concretized with ``f``.
    """
    like = like or qp.math.get_interface(x)

    if like == "jax":
        import jax.core
        from jax.extend.core import concrete_or_error

        if isinstance(x, jax.core.Tracer):
            return concrete_or_error(f, x, context=context)

    return f(x) if f is not None else x


def can_replace(x: TensorLike, y: TensorLike) -> bool:
    r"""Returns True if ``x`` can safely be replaced with ``y`` in a computation."""
    return not qp.math.requires_grad(x) and qp.math.allclose(x, y)
