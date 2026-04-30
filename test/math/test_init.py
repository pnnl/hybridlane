# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pytest
from pennylane import math as pl_math

from hybridlane import math


@pytest.mark.unit
def test_fallback_to_pennylane():
    # Some examples of functions we haven't overriden
    assert math.norm == pl_math.norm
    assert math.choi_matrix == pl_math.choi_matrix


@pytest.mark.unit
def test_dir():
    funcs = set(dir(math))
    assert "pl_math" not in funcs
