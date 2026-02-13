from doctest import ELLIPSIS, NORMALIZE_WHITESPACE

import numpy as base_numpy
import pennylane as qml
import scipy as base_scipy
from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser

import hybridlane as hqml

try:
    import jax
except ImportError:
    jax = None

try:
    import torch
except ImportError:
    torch = None


namespace = {
    "hqml": hqml,
    "qml": qml,
    "np": base_numpy,
    "sp": base_scipy,
    "pnp": qml.numpy,
    "jax": jax,
    "torch": torch,
    "jnp": getattr(jax, "numpy", None),
}


def reset_pennylane_state(namespace):
    qml.capture.disable()
    qml.decomposition.disable_graph()

    if jax:
        jax.config.update("jax_dynamic_shapes", False)


pytest_collect_file = Sybil(
    setup=lambda ns: ns.update(namespace),
    parsers=[
        DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE),
        PythonCodeBlockParser(),
    ],
    patterns=["docs/source/*.rst", "*.py"],
    teardown=reset_pennylane_state,
).pytest()
