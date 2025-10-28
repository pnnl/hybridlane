# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
r"""Module defining static analysis passes for type-checking circuits"""

from .base import (
    BasisSchema,
    ComputationalBasis,
    Qubit,
    Qumode,
    StaticAnalysisResult,
    WireType,
)
from .exceptions import StaticAnalysisError
from .infer_wires import (
    analyze,
    infer_schema_from_observable,
    infer_schema_from_tensors,
)

__all__ = [
    "BasisSchema",
    "StaticAnalysisResult",
    "ComputationalBasis",
    "StaticAnalysisError",
    "WireType",
    "Qubit",
    "Qumode",
    "analyze",
    "infer_schema_from_observable",
    "infer_schema_from_tensors",
]
