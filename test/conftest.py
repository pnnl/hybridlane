# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
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
