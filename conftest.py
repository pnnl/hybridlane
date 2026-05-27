# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
# Based on Sybil documentation: https://sybil.readthedocs.io/en/latest/quickstart.html
from doctest import ELLIPSIS
from typing import Any

import numpy as np
import pennylane as qp
import pytest
from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser

import hybridlane as hl

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

printoptions = np.get_printoptions()


def setup(namespace: dict[str, Any]):
    namespace |= {"qp": qp, "hl": hl, "np": np, "jnp": jnp}
    qp.decomposition.enable_graph()
    np.set_printoptions(precision=4, suppress=True)
    if jax:
        jax.config.update("jax_enable_x64", True)


def teardown(namespace: dict[str, Any]):
    np.set_printoptions(**printoptions)


pytest_collect_file = Sybil(
    parsers=[
        DocTestParser(optionflags=ELLIPSIS),
        PythonCodeBlockParser(),
    ],
    patterns=["docs/source/*.rst", "*.py"],
    setup=setup,
    teardown=teardown,
    name="sybil",
).pytest()


def pytest_collection_modifyitems(items):
    for item in items:
        if "sybil" in item.nodeid:
            item.add_marker(pytest.mark.docs)
