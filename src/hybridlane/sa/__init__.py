r"""Module defining static analysis passes for type-checking circuits"""

from .base import BasisSchema, ComputationalBasis, StaticAnalysisResult
from .exceptions import StaticAnalysisError
from .infer_wires import (
    infer_schema_from_observable,
    infer_schema_from_tensors,
    infer_wire_types,
    analyze,
)

__all__ = [
    "BasisSchema",
    "StaticAnalysisResult",
    "ComputationalBasis",
    "StaticAnalysisError",
    "analyze",
    "infer_schema_from_observable",
    "infer_schema_from_tensors",
    "infer_wire_types",
]
