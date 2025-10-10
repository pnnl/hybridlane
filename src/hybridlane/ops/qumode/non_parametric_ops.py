# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math
from collections.abc import Sequence

import pennylane as qml
from pennylane.decomposition.symbolic_decomposition import (
    make_pow_decomp_with_period,
    pow_involutory,
    self_adjoint,
)
from pennylane.operation import CVOperation
from pennylane.wires import WiresLike

from .parametric_ops_multi_qumode import Beamsplitter
from .parametric_ops_single_qumode import Rotation


class Fourier(CVOperation):
    r"""Continuous-variable Fourier gate :math:`F`

    This gate is a special case of the CV :py:class:`~hybridlane.Rotation` gate with :math:`\theta = \pi/2`
    """

    num_params = 0
    num_wires = 1

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    @staticmethod
    def compute_decomposition(
        *params, wires, **hyperparameters
    ) -> Sequence[CVOperation]:
        return [Rotation(math.pi / 2, wires)]

    def adjoint(self):
        return Rotation(-math.pi / 2, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "F", cache=cache
        )


@qml.register_resources({Rotation: 1})
def _f_to_r(wires, **_):
    Rotation(math.pi / 2, wires)


@qml.register_resources({Rotation: 1})
def _adjoint_f_to_r(wires, **_):
    Rotation(-math.pi / 2, wires)


@qml.register_resources({Rotation: 1})
def _pow_f_to_r(wires, z, **_):
    Rotation(math.pi / 2 * z, wires)


qml.add_decomps(Fourier, _f_to_r)
qml.add_decomps("Adjoint(Fourier)", _adjoint_f_to_r)
qml.add_decomps("Pow(Fourier)", make_pow_decomp_with_period(4), _pow_f_to_r)


class ModeSwap(CVOperation):
    r"""Continuous-variable SWAP between two qumodes

    This has a decomposition in terms of a :py:class:`~hybridlane.Beamsplitter` and phase-space
    :py:class:`~hybridlane.Rotation` gates to eliminate the global phase. See eq. 175 [1]_.

    .. [1] Y. Liu et al, 2024. `arXiv:2407.10381 <https://arxiv.org/abs/2407.10381>`_
    """

    num_params = 0
    num_wires = 2

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    @staticmethod
    def compute_decomposition(*params, wires, **hyperparameters):
        return [
            Beamsplitter(math.pi, 0, wires),
            Rotation(-math.pi / 2, wires[0]),
            Rotation(-math.pi / 2, wires[1]),
        ]

    def adjoint(self):
        return ModeSwap(self.wires)  # self-adjoint up to a global phase of -1

    def pow(self, z: int | float):
        if isinstance(z, float):
            raise NotImplementedError("Unknown formula for fractional powers")
        elif z < 0:
            raise NotImplementedError("Unknown formula for inverse")

        if z % 2 == 0:
            return [qml.Identity(self.wires)]
        else:
            return [ModeSwap(self.wires)]


@qml.register_resources({Beamsplitter: 1, Rotation: 2})
def _swap_to_bs(wires, **_):
    Beamsplitter(math.pi, 0, wires)
    Rotation(-math.pi / 2, wires[0])
    Rotation(-math.pi / 2, wires[1])


qml.add_decomps(ModeSwap, _swap_to_bs)
qml.add_decomps("Adjoint(ModeSwap)", self_adjoint)
qml.add_decomps("Pow(ModeSwap)", pow_involutory)
