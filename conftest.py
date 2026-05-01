# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
# Based on Sybil documentation: https://sybil.readthedocs.io/en/latest/quickstart.html
import importlib.util
from doctest import ELLIPSIS
from typing import Any

import numpy as np
import pennylane as qml
from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser

import hybridlane as hqml


def setup(namespace: dict[str, Any]):
    namespace |= {
        "qml": qml,
        "hqml": hqml,
        "np": np,
    }


# todo: rework doc testing to skip tests involving bosonic qiskit if it's not installed
if importlib.util.find_spec("bosonic_qiskit") is not None:
    pytest_collect_file = Sybil(
        parsers=[
            DocTestParser(optionflags=ELLIPSIS),
            PythonCodeBlockParser(),
        ],
        patterns=["docs/source/*.rst", "*.py"],
        setup=setup,
    ).pytest()
