# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

import math
from typing import cast

import pennylane as qml
from pennylane.ops import Operation
from pennylane.wires import WiresLike

from ..ops import Blue, Hybrid, Red


class FockState(Operation, Hybrid):
    r"""Prepares a definite Fock state from the vacuum

    Unlike PennyLane's :class:`~pennylane.ops.cv.FockState`, this class uses a sequence
    of :py:class:`~hybridlane.ops.Red` and :py:class:`~hybridlane.ops.Blue`
    gates, requiring an ancilla qubit.

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    This prepares a definite Fock state on a qumode using a sequence of red and blue
    sideband gates, favoring the Sideband ISA :footcite:p:`liu2026hybrid`. The gate
    sequence to prepare Fock state :math:`\ket{n}` is given by

    .. math::

        X^{n\mod 2} JC(\frac{\pi}{2\sqrt{n+1}}, \frac{\pi}{2})
            AJC(\frac{\pi}{2\sqrt{n}}, \frac{\pi}{2}) \dots
            JC(\frac{\pi}{2\sqrt{2}}, \frac{\pi}{2}) AJC(\frac{\pi}{2}, \frac{\pi}{2})

    The final :math:`X` gate is applied if :math:`n` is odd to uncompute the qubit.

    This also provides a decomposition for PennyLane's
    :class:`~pennylane.ops.cv.FockState` that uses an ancilla qubit to prepare the Fock state on the qumode, requiring dynamic qubit allocation.

    References
    ----------

    .. footbibliography::
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


def _fockstate_resources(fock_level):
    return {
        Blue: math.ceil(fock_level / 2),
        Red: math.floor(fock_level / 2),
        qml.X: fock_level % 2,
    }


@qml.register_resources(_fockstate_resources)
def _fockstate_decomp(*params, wires, **_):
    fock_state = cast(int, params[0])
    for n in range(fock_state):
        rabi_rate = math.sqrt(n + 1)
        theta = math.pi / (2 * rabi_rate)
        if n % 2 == 0:
            Blue(theta, math.pi / 2, wires)
        else:
            Red(theta, math.pi / 2, wires)

    if fock_state % 2 == 1:
        qml.X(wires[0])


@qml.register_resources({FockState: 1}, work_wires={"zeroed": 1})
def _qml_fockstate_with_ancilla_qubit(n, wires):
    with qml.allocate(1, "zero", restored=True) as ancilla:
        FockState(n, wires=[ancilla[0], wires[0]])


qml.add_decomps(FockState, _fockstate_decomp)
qml.add_decomps(qml.FockState, _qml_fockstate_with_ancilla_qubit)
