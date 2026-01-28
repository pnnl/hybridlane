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

import hybridlane as hqml

from ..op_math.decompositions.qubit_conditioned_decompositions import to_native_qcond


class Fourier(CVOperation):
    r"""Continuous-variable Fourier gate :math:`F`

    This gate is a special case of the CV :py:class:`~hybridlane.Rotation` gate with
    :math:`\theta = \pi/2`
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
        return [hqml.Rotation(math.pi / 2, wires)]

    def adjoint(self):
        return hqml.Rotation(-math.pi / 2, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "F", cache=cache
        )


def _f_to_r_resources():
    return {hqml.Rotation: 1}


@qml.register_resources(_f_to_r_resources)
def _f_to_r(wires, **_):
    hqml.Rotation(math.pi / 2, wires)


def _adjoint_f_to_r_resources():
    return {hqml.Rotation: 1}


@qml.register_resources(_adjoint_f_to_r_resources)
def _adjoint_f_to_r(wires, **_):
    hqml.Rotation(-math.pi / 2, wires)


def _pow_f_to_r_resources(z, **_):
    return {hqml.Rotation: 1}


@qml.register_resources(_pow_f_to_r_resources)
def _pow_f_to_r(wires, z, **_):
    hqml.Rotation(math.pi / 2 * z, wires)


qml.add_decomps(Fourier, _f_to_r)
qml.add_decomps("Adjoint(Fourier)", _adjoint_f_to_r)
qml.add_decomps("Pow(Fourier)", make_pow_decomp_with_period(4), _pow_f_to_r)
qml.add_decomps("qCond(Fourier)", to_native_qcond(1))

F = Fourier
r"""Fourier gate

.. math::

    F = e^{-i\frac{\pi}{2}\hat{n}}

.. seealso::

    This is an alias of :class:`~hybridlane.JaynesCummings`
"""


class ModeSwap(CVOperation):
    r"""Continuous-variable SWAP between two qumodes

    This has a decomposition in terms of a :py:class:`~hybridlane.Beamsplitter` and
    phase-space :py:class:`~hybridlane.Rotation` gates to eliminate the global phase.
    (eq. 175 of :footcite:p:`liu2026hybrid`).

    References
    ----------

    .. footbibliography::
    """

    num_params = 0
    num_wires = 2

    resource_keys = set()

    def __init__(self, wires: WiresLike, id: str | None = None):
        super().__init__(wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

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


def _swap_to_bs_resources():
    return {
        hqml.Beamsplitter: 1,
        hqml.Rotation: 2,
    }


@qml.register_resources(_swap_to_bs_resources)
def _swap_to_bs(wires, **_):
    hqml.Beamsplitter(math.pi, 0, wires)
    hqml.Rotation(-math.pi / 2, wires[0])
    hqml.Rotation(-math.pi / 2, wires[1])


qml.add_decomps(ModeSwap, _swap_to_bs)
qml.add_decomps("Adjoint(ModeSwap)", self_adjoint)
qml.add_decomps("Pow(ModeSwap)", pow_involutory)
