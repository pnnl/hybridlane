# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import math
from typing import cast

import pennylane as qml
from pennylane.ops import Operation
from pennylane.wires import Wires, WiresLike

from ..ops import Blue, Hybrid, Red


class FockLadder(Operation, Hybrid):
    r"""Prepares a definite Fock state from the vacuum

    Unlike :class:`~pennylane.ops.cv.FockState`, this class uses a sequence of
    :py:class:`~hybridlane.ops.Red` and :py:class:`~hybridlane.ops.Blue`
    gates, requiring an ancilla qubit.
    """

    num_params = 1
    num_wires = 2
    num_qumodes = 1
    grad_method = None

    resource_keys = {"fock_level"}

    def __init__(self, n: int, wires: WiresLike = None, id: str | None = None):
        super().__init__(n, wires=wires, id=id)

    @property
    def resource_params(self):
        n = cast(int, self.parameters[0])
        return {"fock_level": n}

    @staticmethod
    def compute_decomposition(*params, wires: Wires, **_):
        fock_state = cast(int, params[0])
        decomp = []
        for n in range(fock_state):
            rabi_rate = math.sqrt(n + 1)
            theta = math.pi / (2 * rabi_rate)
            if n % 2 == 0:
                decomp.append(Blue(theta, math.pi / 2, wires))
            else:
                decomp.append(Red(theta, math.pi / 2, wires))

        return decomp


def _fockladder_resources(fock_level):
    return {Blue: math.ceil(fock_level / 2), Red: math.floor(fock_level / 2)}


@qml.register_resources(_fockladder_resources)
def _fockladder_decomp(*params, wires, **_):
    fock_state = cast(int, params[0])
    for n in range(fock_state):
        rabi_rate = math.sqrt(n + 1)
        theta = math.pi / (2 * rabi_rate)
        if n % 2 == 0:
            Blue(theta, math.pi / 2, wires)
        else:
            Red(theta, math.pi / 2, wires)


qml.add_decomps(FockLadder, _fockladder_decomp)
