# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

__all__ = ["SqueezedCatState"]

from typing import Literal

import numpy as np
import pennylane as qml
from pennylane.decomposition import adjoint_resource_rep
from pennylane.resource import AlgorithmicError, ErrorOperation, SpectralNormError
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

import hybridlane as hqml

from ..ops import Hybrid


class SqueezedCatState(ErrorOperation, Hybrid):
    r"""Cat-state preparation on a qumode using non-Abelian QSP

    This prepares a cat state on a qumode using an ancilliary qubit and hybrid gates,
    differing from the qumode-only operation in Pennylane. Following the protocol of
    :footcite:p:`singh2025towards`, this prepares the cat state

    .. math::

        \ket{\mathcal{C_{\pm\alpha}}} \propto \ket{\alpha_\Delta} \pm
            \ket{-\alpha_\Delta}

    where :math:`\alpha = ae^{i\phi}` and :math:`\ket{\alpha_\Delta}` is a squeezed
    coherent state. This requires :math:`\alpha > 1`, with :math:`\Delta = 1`
    corresponding to no squeezing.

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 3
    * Number of dimensions per parameter: (0, 0, 0)

    The protocol consists of a squeezing gate to optionally squeeze the input state,
    then an ``xCD`` gate to create the superposition of coherent states. Finally, the
    disentangling gadget :math:`\mathcal{U}(\theta', |\alpha|, \Delta)` is applied to
    disentangle the qubit from the qumode.

    **Example:**

    .. code:: python

        qml.decomposition.enable_graph()
        fock_level = 256
        dev = qml.device("bosonicqiskit.hybrid", max_fock_level=fock_level)

        @qml.qnode(dev)
        def circuit(alpha):
            SqueezedCatState(alpha, 0, wires=["q", "m"])

            qml.H("a")
            hqml.ConditionalParity(["a", "m"])
            qml.H("a")
            return hqml.expval(qml.Z("a"))

        alpha = 4
        parity = circuit(alpha)
        assert parity >= 0.95

    Due to the use of the approximate GCR pulse sequence, we can also quantify the
    error in the circuit

    .. code:: python

        errors_dict = qml.resource.algo_error(circuit)(alpha)
        error = errors_dict["SpectralNormError"]
        print(error)

    References
    ----------

    .. footbibliography::
    """

    num_wires = 2
    num_params = 3  # pyright: ignore[reportIncompatibleMethodOverride]
    type_signature = (hqml.sa.Qubit(), hqml.sa.Qumode())
    grad_method = None  # pyright: ignore[reportIncompatibleMethodOverride]

    resource_keys = {"parity"}

    def __init__(
        self,
        a: TensorLike,
        phi: TensorLike,
        delta: TensorLike = 1,
        parity: Literal["even", "odd"] = "even",
        wires: WiresLike = None,
        id: str | None = None,
    ):
        r"""
        Args:
            a: Magnitude of the displacement

            phi: Direction of the displacement

            delta: Uncertainty in the resulting squeezed state. A value of
                :math:`\Delta = 1` has no squeezing, :math:`\Delta < 1` squeeezes in
                position, and :math:`\Delta > 1` anti-squeezes in position. To achieve
                the desired squeezing, a squeezing gate :math:`S(r)` with
                :math:`r = -\log\Delta` is applied.

            parity: Whether to produce a cat state with even or odd photon number
                parity. Must be one of ``even`` or ``odd``.

            wires: The wires this gate acts on

            id: An optional identifier for the gate
        """
        super().__init__(a, phi, delta, wires=wires, id=id)

        self.hyperparameters["parity"] = parity

    def error(self) -> AlgorithmicError:
        return GCR(-np.pi / 2, self.data[0], self.data[2], wires=self.wires).error()


def _squeezedcatstate_resources(parity):
    res = {hqml.XCD: 1, adjoint_resource_rep(GCR): 1, hqml.Squeezing: 1, qml.S: 2}

    if parity == "odd":
        res[qml.X] = 1

    return res


@qml.register_resources(_squeezedcatstate_resources)
def _squeezedcatstate_decomp(a, phi, delta, wires, **hyperparameters):
    if hyperparameters["parity"] == "odd":
        qml.X(wires[0])

    hqml.Squeezing(-qml.math.log(delta), 0, wires[1])
    hqml.XCD(a, phi, wires)
    qml.adjoint(qml.S)(wires[0])
    qml.adjoint(GCR)(-np.pi / 2, a, delta, wires)
    qml.S(wires[0])


qml.add_decomps(SqueezedCatState, _squeezedcatstate_decomp)


class GaussianControlledRotation(ErrorOperation, Hybrid):
    r"""Gaussian-controlled rotation (GCR) pulse sequence

    This is a helper gate for the other templates. It implements the GCR pulse sequence
    of :footcite:p:`singh2025towards`.

    References
    ----------

    .. footbibliography::
    """

    num_wires = 2
    type_signature = (hqml.sa.Qubit(), hqml.sa.Qumode())

    num_params = 3
    ndim_params = (0, 0, 0)

    def __init__(
        self,
        theta: TensorLike,
        a: TensorLike,
        delta: TensorLike = 1,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(theta, a, delta, wires=wires, id=id)

    def error(self) -> AlgorithmicError:
        _, a, delta = self.parameters
        # This is defined for GCR(2θ)
        chi = np.pi * delta / (4 * qml.math.abs(a))
        # eq. c44 of singh et al
        p_error = (5 * chi**6 - 5 * chi**8 / 96) / (1 - 29 * chi**8 / 768)
        return SpectralNormError(p_error)


@qml.register_resources({hqml.YCD: 1, hqml.XCD: 1})
def _gcr_decomp(theta, a, delta, wires, **_):
    hqml.YCD(theta * delta**2 / (4 * qml.math.abs(a)), np.pi, wires)
    hqml.XCD(theta / (4 * qml.math.abs(a)), np.pi / 2, wires)


qml.add_decomps(GaussianControlledRotation, _gcr_decomp)

GCR = GaussianControlledRotation
