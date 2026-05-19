# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import re

import numpy as np
import pennylane as qp
import pytest

import hybridlane as hl


def evaluate_openqasm_compliance(s: str):
    from openqasm3.parser import parse

    parse(s)  # errors if there's syntax mistake


@pytest.mark.bq
@pytest.mark.integration
class TestCircuits:
    @pytest.mark.parametrize("strict", (True, False))
    def test_with_nondiagonal_measurement(self, strict):
        dev = qp.device("bosonicqiskit.hybrid")

        @qp.qnode(dev)
        def circuit(n):
            for j in range(n):
                qp.X(0)
                hl.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, 1])

            return (
                hl.var(hl.QuadP(1)),
                hl.expval(qp.PauliZ(0)),
            )

        qasm = hl.to_openqasm(circuit, precision=5, strict=strict)(5)

        p = re.compile(r"cv_jc")
        assert len(p.findall(qasm)) == 5

        p = re.compile(r"state_prep\(\);")
        assert len(p.findall(qasm)) == 1

        assert "bit[1] c1;" in qasm

        if strict:
            evaluate_openqasm_compliance(qasm)
        else:
            assert "qubit[1] q;" in qasm
            assert "qumode[1] m;" in qasm

            assert "float c0 = measure_x m[0];" in qasm
            assert "c1[0] = measure q[0];" in qasm

    @pytest.mark.parametrize("strict", (True, False))
    def test_with_noncommuting_measurements(self, strict):
        dev = qp.device("bosonicqiskit.hybrid")

        @qp.qnode(dev)
        def circuit(n):
            for j in range(n):
                qp.X(0)
                hl.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, 1])
                hl.SelectiveNumberArbitraryPhase(0.5, j, 1)

            return (
                hl.expval(hl.NumberOperator(1)),
                hl.expval(qp.PauliZ(0)),
                hl.expval(hl.QuadP(1)),  # should be diagonalized
            )

        qasm = hl.to_openqasm(circuit, precision=5, strict=strict)(5)

        p = re.compile(r"cv_snap")
        assert len(p.findall(qasm)) == 5
        p = re.compile(r"cv_jc")
        assert len(p.findall(qasm)) == 5

        # Because n and p don't commute, we need 2 repetitions of the circuit
        p = re.compile(r"state_prep\(\);")
        assert len(p.findall(qasm)) == 2

        assert "bit[1] c1;" in qasm

        if strict:
            evaluate_openqasm_compliance(qasm)
        else:
            assert "qubit[1] q;" in qasm
            assert "qumode[1] m;" in qasm

            assert "uint c0 = measure_n m[0];" in qasm
            assert "c1[0] = measure q[0];" in qasm
            assert "float c2 = measure_x m[0];" in qasm

    @pytest.mark.parametrize("strict", (True, False))
    def test_with_pennylane_gate(self, strict):
        dev = qp.device("bosonicqiskit.hybrid")

        @qp.qnode(dev)
        def circuit():
            qp.X(0)
            qp.Beamsplitter(np.pi / 2, 0, [1, 2])
            qp.Kerr(5.0, 1)

            return (
                qp.var(hl.QuadP(1)),
                qp.expval(qp.PauliZ(0)),
            )

        qasm = hl.to_openqasm(circuit, precision=5, strict=strict)()

        # Beamsplitter gets converted
        assert "cv_bs(3.14159, -1.57080) m[0], m[1];" in qasm
        assert "cv_k(-5.00000) m[0];" in qasm

        p = re.compile(r"state_prep\(\);")
        assert len(p.findall(qasm)) == 1

        assert "bit[1] c1;" in qasm
        assert "cv_r(1.57080) m[0];" in qasm

        if strict:
            evaluate_openqasm_compliance(qasm)
        else:
            assert "qubit[1] q;" in qasm
            assert "qumode[2] m;" in qasm

            assert "float c0 = measure_x m[0];" in qasm
            assert "c1[0] = measure q[0];" in qasm
