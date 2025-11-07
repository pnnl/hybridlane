import numpy as np
import pennylane as qml
from pennylane.wires import Wires

import hybridlane as hqml


class TestGaussianFusion:
    def test_displacement(self):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @hqml.transforms.gaussian_fusion
        @qml.qnode(dev)
        def circuit():
            hqml.Displacement(0.5, np.pi / 4, 0)
            hqml.Displacement(0.5, np.pi / 4, 1)
            hqml.Displacement(0.5, np.pi / 4, 0)
            hqml.Displacement(0.5, np.pi / 4, 1)

        tape = qml.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 1
        assert isinstance(tape.operations[0], hqml.Gaussian)
        assert tape.wires == Wires([0, 1])

        S = tape.operations[0].parameters[0]
        assert S.shape == (5, 5)
        assert hqml.heisenberg.is_symplectic(S)
        assert np.allclose(S[1:, 0], np.ones(4) / np.sqrt(2))
        assert np.allclose(S[1:, 1:], np.eye(4))

    def test_with_nongaussian(self):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @hqml.transforms.gaussian_fusion
        @qml.qnode(dev)
        def circuit():
            hqml.Displacement(0.5, np.pi / 4, 0)
            hqml.Displacement(0.5, np.pi / 4, 1)
            hqml.Kerr(0.5, 0)
            hqml.Displacement(0.5, np.pi / 4, 0)
            hqml.Displacement(0.5, np.pi / 4, 1)

        tape = qml.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 3
        assert isinstance(tape.operations[0], hqml.Gaussian)
        assert isinstance(tape.operations[1], hqml.Kerr)
        assert isinstance(tape.operations[2], hqml.Gaussian)
        assert tape.wires == Wires([0, 1])

    def test_with_nongaussian_commuting(self):
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @hqml.transforms.gaussian_fusion
        @qml.qnode(dev)
        def circuit():
            hqml.Displacement(0.5, np.pi / 4, 0)
            hqml.Displacement(0.5, np.pi / 4, 1)
            hqml.Kerr(0.5, 2)
            hqml.Displacement(0.5, np.pi / 4, 0)
            hqml.Displacement(0.5, np.pi / 4, 1)

        tape = qml.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 2
        assert isinstance(tape.operations[0], hqml.Kerr)
        assert isinstance(tape.operations[1], hqml.Gaussian)
        assert tape.wires == Wires([2, 0, 1])

    def test_large_random_gaussian(self):
        gaussians = [
            hqml.Displacement,
            hqml.Rotation,
            hqml.Squeezing,
            hqml.TwoModeSqueezing,
            hqml.TwoModeSum,
            hqml.Beamsplitter,
        ]

        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=8)

        @hqml.transforms.gaussian_fusion
        @qml.qnode(dev)
        def circuit():
            rng = np.random.default_rng()
            for _ in range(100):
                op_type = rng.choice(gaussians)
                wires = rng.choice(5, size=op_type.num_wires, replace=False)
                params = rng.normal(size=op_type.num_params)
                op_type(*params, wires)

        tape = qml.workflow.construct_tape(circuit)()
        assert len(tape.operations) == 1
        assert isinstance(tape.operations[0], hqml.Gaussian)
