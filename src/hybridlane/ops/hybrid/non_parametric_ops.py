# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math

import pennylane as qml
from pennylane import numpy as np
from pennylane.decomposition.symbolic_decomposition import (
    make_pow_decomp_with_period,
)
from pennylane.operation import Operation
from pennylane.wires import WiresLike

import hybridlane as hqml

from ..mixins import Hybrid
from ..op_math.decompositions.qubit_conditioned_decompositions import (
    decompose_multiqcond_native,
)


class ConditionalParity(Operation, Hybrid):
    r"""Qubit-conditioned number parity gate :math:`CP`

    This gate is a special case of the :py:class:`~hybridlane.ConditionalRotation`
    gate, with :math:`CP = CR(\pi)`, resulting in the unitary expression

    .. math::

        CP &= \exp[-i\frac{\pi}{2}\sigma_z \hat{n}] \\
           &= \ket{0}\bra{0} \otimes F + \ket{1}\bra{1} \otimes F^\dagger

    This gate can also be viewed as the "conditioned" version of the
    :class:`~hybridlane.Fourier` gate.

    .. seealso::

        :class:`~hybridlane.ConditionalRotation`
    """

    num_params = 0
    num_wires = 2
    num_qumodes = 1

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    @staticmethod
    def compute_decomposition(wires, **_):
        return [hqml.ConditionalRotation(math.pi, wires)]

    def adjoint(self):
        return hqml.ConditionalRotation(-math.pi, self.wires)

    def pow(self, z: int | float) -> list[Operation]:
        z_mod4 = z % 4

        if np.allclose(z_mod4, 0):
            return []

        return [hqml.ConditionalRotation(math.pi * z_mod4, self.wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CΠ", cache=cache
        )


def _cp_resources(**_):
    return {hqml.ConditionalRotation: 1}


@qml.register_resources(_cp_resources)
def _cp_to_cr(wires, **_):
    hqml.ConditionalRotation(math.pi, wires)


@qml.register_resources(_cp_resources)
def _adjoint_cp_to_cr(wires, **_):
    hqml.ConditionalRotation(-math.pi, wires)


@qml.register_resources(_cp_resources)
def _pow_cp_to_cr(wires, z, **_):
    z_mod4 = z % 4
    hqml.ConditionalRotation(math.pi * z_mod4, wires=wires)


qml.add_decomps(ConditionalParity, _cp_to_cr)
qml.add_decomps("Adjoint(ConditionalParity)", _adjoint_cp_to_cr)
qml.add_decomps("Pow(ConditionalParity)", make_pow_decomp_with_period(4), _pow_cp_to_cr)
qml.add_decomps("qCond(ConditionalParity)", decompose_multiqcond_native)
