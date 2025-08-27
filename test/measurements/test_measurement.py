import numpy as np
import pytest
from pennylane.wires import Wires

from hybridlane.measurements import (
    CountsResult,
    FockTruncation,
    SampleResult,
)
from hybridlane.sa.base import BasisSchema, ComputationalBasis


class TestBasisSchema:
    def test_init(self):
        wire_map = {
            Wires("a"): ComputationalBasis.Discrete,
            Wires("b"): ComputationalBasis.Position,
        }
        schema = BasisSchema(wire_map)
        assert schema.get_basis("a") == ComputationalBasis.Discrete
        assert schema.get_basis("b") == ComputationalBasis.Position

    def test_multi_init(self):
        wire_map = {
            Wires("a"): ComputationalBasis.Discrete,
            Wires(["b", "c"]): ComputationalBasis.Position,
        }
        schema = BasisSchema(wire_map)
        assert schema.get_basis("a") == ComputationalBasis.Discrete
        assert schema.get_basis("b") == ComputationalBasis.Position
        assert schema.get_basis("c") == ComputationalBasis.Position

    def test_init_error(self):
        with pytest.raises(ValueError):
            BasisSchema({Wires("a"): "not a basis"})

    def test_eq(self):
        schema1 = BasisSchema({Wires("a"): ComputationalBasis.Discrete})
        schema2 = BasisSchema({Wires("a"): ComputationalBasis.Discrete})
        assert schema1 == schema2

    def test_neq(self):
        schema1 = BasisSchema({Wires("a"): ComputationalBasis.Discrete})
        schema2 = BasisSchema({Wires("b"): ComputationalBasis.Discrete})
        assert schema1 != schema2


class TestSampleResult:
    def test_init_basis_states(self):
        basis_states = {"a": np.array([1, 0]), "b": np.array([0, 1])}
        result = SampleResult(basis_states=basis_states)
        assert result.is_basis_states
        assert not result.is_eigvals
        assert result.shots == 2

    def test_init_eigvals(self):
        eigvals = np.array([1.0, -1.0])
        result = SampleResult(eigvals=eigvals)
        assert not result.is_basis_states
        assert result.is_eigvals
        assert result.shots == 2

    def test_init_error(self):
        with pytest.raises(ValueError):
            SampleResult()

    def test_concatenate(self):
        basis_states1 = {"a": np.array([1]), "b": np.array([0])}
        result1 = SampleResult(basis_states=basis_states1)
        basis_states2 = {"a": np.array([0]), "b": np.array([1])}
        result2 = SampleResult(basis_states=basis_states2)
        result3 = result1.concatenate(result2)
        assert result3.shots == 2


class TestCountsResult:
    def test_init_basis_states(self):
        counts = {(0, 1): 10, (1, 0): 20}
        wire_order = Wires(["a", "b"])
        basis_schema = BasisSchema(
            {
                Wires("a"): ComputationalBasis.Discrete,
                Wires("b"): ComputationalBasis.Discrete,
            }
        )
        result = CountsResult(
            counts=counts, wire_order=wire_order, basis_schema=basis_schema
        )
        assert result.is_basis_states
        assert not result.is_eigenvals
        assert result.shots == 30

    def test_init_eigvals(self):
        counts = {1.0: 15, -1.0: 25}
        result = CountsResult(counts=counts)
        assert not result.is_basis_states
        assert result.is_eigenvals
        assert result.shots == 40


class TestFockTruncation:
    def test_shape(self):
        schema = BasisSchema(
            {
                Wires("a"): ComputationalBasis.Discrete,
                Wires("b"): ComputationalBasis.Position,
            }
        )
        truncation = FockTruncation(basis_schema=schema, dim_sizes={"a": 2, "b": 10})
        shape = truncation.shape(Wires(["a", "b"]))
        assert shape == (2, 10)

        shape = truncation.shape(Wires(["b", "a"]))
        assert shape == (10, 2)
