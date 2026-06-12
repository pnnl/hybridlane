# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import pytest
from pennylane.exceptions import MeasurementShapeError
from pennylane.wires import Wires

import hybridlane as hl
from hybridlane.measurements import (
    CountsResult,
    FockTruncation,
    SampleResult,
)
from hybridlane.sa.base import BasisSchema, ComputationalBasis


@pytest.mark.unit
class TestBasisSchema:
    def test_init(self):
        wire_map = {
            "a": ComputationalBasis.Discrete,
            "b": ComputationalBasis.Position,
        }
        schema = BasisSchema(wire_map)
        assert schema.get_basis("a") == ComputationalBasis.Discrete
        assert schema.get_basis("b") == ComputationalBasis.Position

    def test_multi_init(self):
        wire_map = {
            "a": ComputationalBasis.Discrete,
            ("b", "c"): ComputationalBasis.Position,
        }
        schema = BasisSchema(wire_map)
        assert schema.get_basis("a") == ComputationalBasis.Discrete
        assert schema.get_basis("b") == ComputationalBasis.Position
        assert schema.get_basis("c") == ComputationalBasis.Position

    def test_init_error(self):
        with pytest.raises(ValueError):
            BasisSchema({"a": "not a basis"})

    def test_eq(self):
        schema1 = BasisSchema({"a": ComputationalBasis.Discrete})
        schema2 = BasisSchema({"a": ComputationalBasis.Discrete})
        assert schema1 == schema2

    def test_neq(self):
        schema1 = BasisSchema({"a": ComputationalBasis.Discrete})
        schema2 = BasisSchema({"b": ComputationalBasis.Discrete})
        assert schema1 != schema2


@pytest.mark.unit
@pytest.mark.all_interfaces
class TestSampleResult:
    def test_init_basis_states(self, like):
        basis_states = {
            "a": hl.math.array([1, 0], like=like),
            "b": hl.math.array([0, 1], like=like),
        }
        result = SampleResult.from_basis_states(basis_states)
        assert result.shots == 2

    def test_batch_dim(self, like):
        basis_states = {
            0: hl.math.array([[1, 0]], like=like),
            1: hl.math.array([[0, 1]], like=like),
        }
        result = SampleResult.from_basis_states(basis_states)
        assert result.shots == 2
        assert result.batch_size == 1

    def test_batch_dim_mismatch(self, like):
        basis_states = {
            0: hl.math.array([[1, 0]], like=like),
            1: hl.math.array([[0, 1], [1, 0]], like=like),
        }

        with pytest.raises(MeasurementShapeError):
            SampleResult.from_basis_states(basis_states)

    def test_shot_mismatch(self, like):
        basis_states = {
            0: hl.math.array([1, 0], like=like),
            1: hl.math.array([0, 1, 1], like=like),
        }

        with pytest.raises(MeasurementShapeError):
            SampleResult.from_basis_states(basis_states)

    def test_concatenate(self, like):
        basis_states1 = {
            "a": hl.math.array([1], like=like),
            "b": hl.math.array([0], like=like),
        }
        result1 = SampleResult.from_basis_states(basis_states1)
        basis_states2 = {
            "a": hl.math.array([0], like=like),
            "b": hl.math.array([1], like=like),
        }
        result2 = SampleResult.from_basis_states(basis_states2)
        result3 = result1.concatenate(result2)
        assert result3.shots == 2
        assert result3.batch_size is None


@pytest.mark.unit
class TestCountsResult:
    def test_init_basis_states(self):
        counts = {(0, 1): 10, (1, 0): 20}
        wire_order = Wires(["a", "b"])
        basis_schema = BasisSchema({wire_order: ComputationalBasis.Discrete})
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


@pytest.mark.unit
class TestFockTruncation:
    def test_shape(self):
        schema = BasisSchema(
            {"a": ComputationalBasis.Discrete, "b": ComputationalBasis.Position}
        )
        truncation = FockTruncation(basis_schema=schema, dim_sizes={"a": 2, "b": 10})
        shape = truncation.shape(Wires(["a", "b"]))
        assert shape == (2, 10)

        shape = truncation.shape(Wires(["b", "a"]))
        assert shape == (10, 2)
