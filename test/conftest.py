# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import importlib.util

import pennylane as qp
import pytest


@pytest.fixture(autouse=True)
def enable_graph_decomp():
    qp.decomposition.enable_graph()
    yield
    qp.decomposition.disable_graph()


@pytest.fixture(autouse=True)
def disable_graph_decomp():
    qp.decomposition.disable_graph()
    yield
    qp.decomposition.enable_graph()


def pytest_collection_modifyitems(config, items):
    if importlib.util.find_spec("jax") is None:
        skip_jax = pytest.mark.skip(reason="jax not installed")
        for item in items:
            if "jax" in item.keywords:
                item.add_marker(skip_jax)

    if importlib.util.find_spec("torch") is None:
        skip_torch = pytest.mark.skip(reason="torch not installed")
        for item in items:
            if "torch" in item.keywords:
                item.add_marker(skip_torch)

    if importlib.util.find_spec("bosonic_qiskit") is None:
        skip_bq = pytest.mark.skip(reason="bosonic_qiskit not installed")
        for item in items:
            if "bq" in item.keywords:
                item.add_marker(skip_bq)
