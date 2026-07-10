# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import warnings
from typing import ClassVar

import numpy as np
import pennylane as qp
import pytest
from pennylane.decomposition.symbolic_decomposition import pow_rotation
from pennylane.operation import Operation
from pennylane.templates import QuantumPhaseEstimation
from pennylane.transforms import resolve_dynamic_wires

import hybridlane as hl
from hybridlane.ops.mixins import Hybrid
from hybridlane.wires import BasisMap, ComputationalBasis


# Necessary to define this class so we can give the pow_rotation decomposition. Otherwise,
# pennylane uses the pow_repeat_base decomposition and that dramatically expands the circuit depth
class Evo(Operation, Hybrid):
    num_wires = 2
    num_qumodes = 1
    num_params = 1

    resource_keys: ClassVar = set()

    def __init__(self, t: float, omega_r=1, omega_q=-1, chi=0.1, wires=None, id=None):
        self.hyperparameters.update({"omega_r": omega_r, "omega_q": omega_q, "chi": chi})
        super().__init__(t, wires=wires, id=id)


@qp.register_resources({qp.RZ: 1, hl.Rotation: 1, hl.ConditionalRotation: 1})
def _evo_decomp(t, wires, omega_r, omega_q, chi, **_):
    qp.RZ(-omega_q * t, wires[0])
    hl.Rotation(omega_r * t, wires[1])
    hl.ConditionalRotation(-chi * t, wires)


qp.add_decomps(Evo, _evo_decomp)
qp.add_decomps("Pow(Evo)", pow_rotation)


@pytest.mark.usefixtures("enable_graph_decomp")
class TestApplications:
    @pytest.mark.bq
    @pytest.mark.integration
    def test_dispersive_qpe(self):
        omega_r = 1
        omega_q = -1
        chi = 0.1
        U = Evo(1, omega_r, omega_q, chi, ("q", "m"))  # noqa: N806

        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=8)

        @qp.transforms.decompose(
            gate_set={
                hl.Red,
                hl.Blue,
                hl.ConditionalRotation,
                hl.Rotation,
                qp.RZ,
                qp.CRZ,
                qp.CNOT,
                qp.H,
                qp.ControlledPhaseShift,
            },
        )
        @qp.set_shots(10)
        @qp.qnode(dev)
        def circuit(n_bits: int):
            hl.FockState(4, wires=("q", "m"))
            estimation_wires = range(n_bits)
            QuantumPhaseEstimation(U, estimation_wires=estimation_wires)

            schema = BasisMap({estimation_wires: ComputationalBasis.Discrete})
            return hl.sample(schema=schema)

        # Decomposition raises a warning if it can't find a decomposition
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            specs = qp.specs(circuit, level="device")(5)

        gate_types = specs["resources"].gate_types
        assert gate_types["Hadamard"] == 10  # 2 per estimation bit
        assert gate_types["ConditionalRotation"] == 15  # 2 per CR term, 1 per R term
        assert gate_types["Rotation"] == 5  # 1 per R term


@pytest.mark.usefixtures("enable_graph_decomp")
class TestGateDecompositions:
    @pytest.mark.integration
    @pytest.mark.bq
    def test_snap_to_sqr(self):
        dev = qp.device("bosonicqiskit.hybrid", max_fock_level=32)

        @qp.qnode(dev)
        def circuit():
            hl.Displacement(0.5, 0, 0)
            hl.SNAP(0.5, 1, 0)
            hl.SNAP(0.5, 2, 0)
            hl.SNAP(0.5, 3, 0)
            hl.Displacement(-0.5, 0, 0)
            return hl.expval(hl.X(0))

        expval_with_snap = circuit()

        sqr_circuit = (
            qp.transforms.decompose(gate_set={hl.Displacement, hl.SQR}, num_work_wires=1)
            + resolve_dynamic_wires(min_int=1, allow_resets=False)
        )(circuit)
        expval_with_sqr = sqr_circuit()

        assert np.allclose(expval_with_sqr, expval_with_snap)
