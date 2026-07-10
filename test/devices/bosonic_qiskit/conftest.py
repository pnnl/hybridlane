# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.fspath.dirpath().basename == "bosonic_qiskit":
            item.add_marker(pytest.mark.bq)
