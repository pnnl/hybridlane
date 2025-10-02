import math
from typing import Optional, cast

from pennylane.ops import Operation
from pennylane.wires import Wires, WiresLike

from ..ops import Hybrid, Red, Blue


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

    resource_keys = {"num_gates"}

    def __init__(self, n: int, wires: WiresLike = None, id: Optional[str] = None):
        super().__init__(n, wires=wires, id=id)

    @property
    def resource_params(self):
        n = cast(int, self.parameters[0])
        return {"num_gates": n}

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
