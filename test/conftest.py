# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import importlib.util

import pennylane as qp
import pytest


@pytest.fixture(autouse=True, scope="session")
def use_jax_x64():
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        yield
        jax.config.update("jax_enable_x64", False)
    except ImportError:
        yield


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
    for item in items:
        # Check if this is a parametrized test with 'like' parameter
        if hasattr(item, "callspec") and "like" in item.callspec.params:
            like_value = item.callspec.params["like"]
            if like_value == "jax":
                item.add_marker(pytest.mark.jax)
            elif like_value == "torch":
                item.add_marker(pytest.mark.torch)

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

    if importlib.util.find_spec("jaqalpaq") is None:
        skip_qscout = pytest.mark.skip(reason="jaqalpaq not installed")
        for item in items:
            if "qscout" in item.keywords:
                item.add_marker(skip_qscout)


def pytest_generate_tests(metafunc):
    # Hook to parametrize tests over deep learning interfaces
    if metafunc.definition.get_closest_marker("all_interfaces"):
        if "like" in metafunc.fixturenames:
            like_values = ["numpy", "jax"]
            metafunc.parametrize("like", like_values)
