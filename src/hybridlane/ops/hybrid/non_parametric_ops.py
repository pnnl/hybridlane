# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math

import pennylane as qml
from pennylane import numpy as np
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    make_pow_decomp_with_period,
    pow_rotation,
)
from pennylane.operation import Operation
from pennylane.wires import WiresLike

from ..mixins import Hybrid


class ConditionalParity(Operation, Hybrid):
    r"""Qubit-conditioned number parity gate :math:`CP`

    This gate is a special case of the :py:class:`~hybridlane.ConditionalRotation` gate, with :math:`CP = CR(\pi)`, resulting
    in the unitary expression

    .. math::

        CP &= \exp[-i\frac{\pi}{2}\sigma_z \hat{n}] \\
           &= \ket{0}\bra{0} \otimes F + \ket{1}\bra{1} \otimes F^\dagger

    This gate can also be viewed as the "conditioned" version of the :class:`~hybridlane.Fourier` gate.

    .. seealso::

        :py:class:`~hybridlane.ConditionalRotation`
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
        from .parametric_ops_single_qumode import ConditionalRotation

        return [ConditionalRotation(math.pi, wires)]

    def adjoint(self):
        from .parametric_ops_single_qumode import ConditionalRotation

        return ConditionalRotation(-math.pi, self.wires)

    def pow(self, z: int | float) -> list[Operation]:
        from .parametric_ops_single_qumode import ConditionalRotation

        z_mod4 = z % 4

        if np.allclose(z_mod4, 0):
            return []

        return [ConditionalRotation(math.pi * z_mod4, self.wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CΠ", cache=cache
        )


def _cp_resources():
    from .parametric_ops_single_qumode import ConditionalRotation

    return {ConditionalRotation: 1}


@qml.register_resources(_cp_resources)
def _cp_to_cr(wires, **_):
    from .parametric_ops_single_qumode import ConditionalRotation

    ConditionalRotation(math.pi, wires)


@qml.register_resources(_cp_resources)
def _adjoint_cp_to_cr(wires, **_):
    from .parametric_ops_single_qumode import ConditionalRotation

    ConditionalRotation(-math.pi, wires)


@qml.register_resources(_cp_resources)
def _pow_cp_to_cr(wires, z, **_):
    from .parametric_ops_single_qumode import ConditionalRotation

    z_mod4 = z % 4
    qml.pow(ConditionalRotation(math.pi * z_mod4, wires=wires), z)


qml.add_decomps(ConditionalParity, _cp_to_cr)
qml.add_decomps("Adjoint(ConditionalParity)", _adjoint_cp_to_cr)
qml.add_decomps("Pow(ConditionalParity)", make_pow_decomp_with_period(4), _pow_cp_to_cr)
