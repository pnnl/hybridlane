# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import numpy as np
import pennylane as qml
import re
import pytest

import hybridlane as hqml


def evaluate_openqasm_compliance(s: str):
    from openqasm3.parser import parse

    program = parse(s)  # errors if there's syntax mistake


class TestCircuits:
    @pytest.mark.parametrize("strict", (True, False))
    def test_with_nondiagonal_measurement(self, strict):
        dev = qml.device("bosonicqiskit.hybrid")

        @qml.qnode(dev)
        def circuit(n):
            for j in range(n):
                qml.X(0)
                hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, 1])

            return (
                hqml.var(hqml.QuadP(1)),
                hqml.expval(qml.PauliZ(0)),
            )

        qasm = hqml.to_openqasm(circuit, precision=5, strict=strict)(5)

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

            assert "float[homodyne_precision_bits] c0 = measure_x m[0];" in qasm
            assert "c1[0] = measure q[0];" in qasm

    @pytest.mark.parametrize("strict", (True, False))
    def test_with_noncommuting_measurements(self, strict):
        dev = qml.device("bosonicqiskit.hybrid")

        @qml.qnode(dev)
        def circuit(n):
            for j in range(n):
                qml.X(0)
                hqml.JaynesCummings(np.pi / (2 * np.sqrt(j + 1)), np.pi / 2, [0, 1])
                hqml.SelectiveNumberArbitraryPhase(0.5, j, [0, 1])

            return (
                hqml.expval(hqml.NumberOperator(1)),
                hqml.expval(qml.PauliZ(0)),
                hqml.expval(hqml.QuadP(1)),  # should be diagonalized
            )

        qasm = hqml.to_openqasm(circuit, precision=5, strict=strict)(5)

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

            assert "uint[fock_readout_precision_bits] c0 = measure_n m[0];" in qasm
            assert "c1[0] = measure q[0];" in qasm
            assert "float[homodyne_precision_bits] c2 = measure_x m[0];" in qasm

    @pytest.mark.parametrize("strict", (True, False))
    def test_with_pennylane_gate(self, strict):
        dev = qml.device("bosonicqiskit.hybrid")

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            qml.Beamsplitter(np.pi / 2, 0, [1, 2])
            qml.Kerr(5.0, 1)

            return (
                qml.var(hqml.QuadP(1)),
                qml.expval(qml.PauliZ(0)),
            )

        qasm = hqml.to_openqasm(circuit, precision=5, strict=strict)()

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

            assert "float[homodyne_precision_bits] c0 = measure_x m[0];" in qasm
            assert "c1[0] = measure q[0];" in qasm
