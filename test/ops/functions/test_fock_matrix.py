# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

import pennylane as qp
import pytest
from pennylane.exceptions import TransformError
from pennylane.pauli.pauli_arithmetic import PauliWord
from pennylane.tape.qscript import make_qscript

import hybridlane as hl


@pytest.mark.unit
@pytest.mark.all_interfaces
def test_operator(like):
    op = qp.X(0)
    mat = hl.fock_matrix(op)
    assert mat == pytest.approx(qp.matrix(op))

    op = qp.RX(hl.math.array(0.123, like=like), 0)
    mat = hl.fock_matrix(op)
    assert hl.math.get_interface(mat) == like
    assert mat == pytest.approx(qp.matrix(op))

    op = qp.CNOT((0, 1))
    mat = hl.fock_matrix(op)
    assert mat == pytest.approx(qp.matrix(op))


@pytest.mark.unit
@pytest.mark.all_interfaces
def test_fock_operator(like):
    wire_dims = {0: 2, 1: 4}
    params = hl.math.array([0.123, 0.456], like=like)
    op = hl.CD(*params, wires=(0, 1))

    mat = hl.fock_matrix(op, wire_dims=wire_dims)
    assert hl.math.get_interface(mat) == like
    assert mat == pytest.approx(op.fock_matrix(wire_dims))

    with pytest.raises(
        ValueError, match="`wire_dims` must be specified for the fock_matrix"
    ):
        hl.fock_matrix(op)


@pytest.mark.unit
def test_wire_order_with_operator():
    op = qp.CNOT((0, 1))
    with pytest.raises(TransformError):
        hl.fock_matrix(op, wire_order=(2, 1))


@pytest.mark.unit
def test_wire_dims_with_operator():
    op = qp.CNOT((0, 1))
    with pytest.raises(ValueError):
        hl.fock_matrix(
            op, wire_order=(0, 1, 2)
        )  # no wire dims to use to figure out what dimension 2 has


@pytest.mark.unit
def test_pauli_word():
    op = PauliWord({"a": "X", 2: "Y", 3: "Z"})
    assert hl.fock_matrix(op, wire_order=("a", 2, 3)) == pytest.approx(
        qp.matrix(op, wire_order=("a", 2, 3))
    )


@pytest.mark.unit
@pytest.mark.all_interfaces
def test_callable(like):
    params = hl.math.array([0.123, 0.456], like=like)
    op_fn = hl.ECD

    # Must provide wire order because hl.ECD has 2 wires
    with pytest.raises(ValueError):
        hl.fock_matrix(op_fn)(*params, wires=(0, 1))

    wire_dims = {0: 2, 1: 4}
    wire_order = (0, 1)
    mat = hl.fock_matrix(op_fn, wire_dims=wire_dims, wire_order=wire_order)(
        *params, wires=wire_order
    )
    assert hl.math.get_interface(mat) == like
    assert mat == pytest.approx(
        hl.ECD(*params, wires=wire_order).fock_matrix(wire_dims)
    )

    def test_fn(params):
        qp.X(0)
        hl.CR(*params, wires=(0, 1))
        qp.X(0)

        return hl.expval(hl.N(1))

    # Have to provide a wire order with a callable
    with pytest.raises(
        ValueError, match="`wire_order` must be specified for callables"
    ):
        hl.fock_matrix(test_fn, wire_dims={0: 2, 1: 4})(params[:1])


@pytest.mark.unit
@pytest.mark.all_interfaces
def test_quantum_fn(like):
    def test_fn(params):
        qp.X(0)
        hl.CR(*params, wires=(0, 1))
        qp.X(0)

        return hl.expval(hl.N(1))

    dev = qp.device("default.hybrid", fock_level=8)

    for obj in (test_fn, qp.QNode(test_fn, dev)):
        params = hl.math.array([0.123], like=like)
        wire_dims = {0: 2, 1: 4}
        wire_order = (0, 1)
        mat = hl.fock_matrix(obj, wire_dims=wire_dims, wire_order=wire_order)(params)
        assert hl.math.get_interface(mat) == like

        expected_mat = hl.CR(*params, wires=(0, 1)).adjoint().fock_matrix(wire_dims)
        assert mat == pytest.approx(expected_mat)


@pytest.mark.unit
@pytest.mark.all_interfaces
def test_sequences(like):
    def test_fn(params):
        qp.X(0)
        hl.CR(*params, wires=(0, 1))
        qp.X(0)

        return hl.expval(hl.N(1))

    params = hl.math.array([0.123], like=like)
    tape = make_qscript(test_fn)(params)

    for obj in (tape, tape.operations):
        wire_dims = {0: 2, 1: 4}
        wire_order = (0, 1)
        mat = hl.fock_matrix(obj, wire_dims=wire_dims, wire_order=wire_order)
        assert hl.math.get_interface(mat) == like

        expected_mat = hl.CR(*params, wires=(0, 1)).adjoint().fock_matrix(wire_dims)
        assert mat == pytest.approx(expected_mat)


@pytest.mark.unit
@pytest.mark.all_interfaces
def test_qnode(like):
    dev = qp.device("default.hybrid", fock_level=8)

    @qp.qnode(dev)
    def test_fn(params):
        qp.X(0)
        hl.CR(*params, wires=(0, 1))
        qp.X(0)

        return hl.expval(hl.N(1))

    params = hl.math.array([0.123], like=like)
    wire_dims = {0: 2, 1: 4}
    # need a wire order because device wires is none
    with pytest.raises(ValueError):
        hl.fock_matrix(test_fn, wire_dims=wire_dims)(params)


@pytest.mark.unit
def test_quantum_fn_wire_dims():
    def test_fn(params):
        qp.X(0)
        hl.CR(*params, wires=(0, 1))
        qp.X(0)

        return hl.expval(hl.N(1))

    params = hl.math.array([0.123])
    tape = make_qscript(test_fn)(params)

    with pytest.raises(
        TransformError, match="Quantum function has a qumode but no dimension"
    ):
        hl.fock_matrix(
            tape, wire_dims={0: 2}, wire_order=(0, 1)
        )  # no dimension for qumode 1

    with pytest.raises(TransformError, match="Wire 0 is of type"):
        hl.fock_matrix(
            tape, wire_dims={0: 3, 1: 3}, wire_order=(0, 1)
        )  # incorrect qubit dimension
