# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import numpy as np
import pennylane as qml
from pennylane.decomposition.symbolic_decomposition import (
    make_pow_decomp_with_period,
)
from pennylane.operation import Operation
from pennylane.wires import WiresLike

import hybridlane as hqml

from ..mixins import FockRepresentation, HybridOperation
from ..op_math.decompositions.qubit_conditioned_decompositions import (
    decompose_multiqcond_native,
)


class ConditionalParity(HybridOperation, FockRepresentation):
    r"""Qubit-conditioned number parity gate :math:`CP`

    .. math::

        CP &= \exp[-i\frac{\pi}{2}\sigma_z \hat{n}] \\
           &= \begin{pmatrix} F & 0 \\ 0 & F^\dagger \end{pmatrix}

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 0
    * Number of dimensions per parameter: None

    This gate is a special case of the :py:class:`~hybridlane.ConditionalRotation`
    gate, with :math:`CP = CR(\pi)`. Its representation in the Fock basis can be obtained
    with:

    >>> CP((0, 1)).fock_matrix({0: 2, 1: 2})
    array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j],
           [0.+0.j, 0.+0.j, 1.-0.j, 0.-0.j],
           [0.+0.j, 0.+0.j, 0.-0.j, 0.+1.j]])

    This gate can also be viewed as the "conditioned" version of the
    :class:`~hybridlane.Fourier` gate.

    >>> hqml.qcond(hqml.F(1), control_wires=0)
    ConditionalParity(wires=[0, 1])

    .. seealso::

        :class:`~hybridlane.ConditionalRotation`
    """

    num_params = 0
    num_wires = 2
    num_qumodes = 1

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    def adjoint(self):
        return hqml.ConditionalRotation(-math.pi, self.wires)

    def pow(self, z: int | float) -> list[Operation]:
        z_mod4 = z % 4

        if np.allclose(z_mod4, 0):
            return []

        return [hqml.ConditionalRotation(math.pi * z_mod4, self.wires)]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CP", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...]) -> np.ndarray:
        f = hqml.Fourier.compute_fock_matrix(wire_dims[1:])
        fd = hqml.math.conj(hqml.math.transpose(f))
        return hqml.math.block_diag([f, fd])


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

CP = ConditionalParity
r"""Conditional parity (CP) gate

.. math::

    CP = e^{-i\frac{\pi}{2}\hat{n}Z}

This is an alias for :class:`~hybridlane.ConditionalParity`
"""
