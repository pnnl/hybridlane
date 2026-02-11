# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

__all__ = ["SqueezedCatState", "GKPState"]

import math
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

    where :math:`\ket{\alpha_\Delta}` is a squeezed coherent state. This requires
    :math:`\alpha > 1`, with :math:`\Delta = 1` corresponding to no squeezing.

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
            SqueezedCatState(alpha, np.pi / 2, wires=["q", "m"])

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
    ndim_params = (0, 0, 0)  # pyright: ignore[reportIncompatibleMethodOverride]
    grad_method = None  # pyright: ignore[reportIncompatibleMethodOverride]
    type_signature = (hqml.sa.Qubit(), hqml.sa.Qumode())

    resource_keys = {"parity", "has_squeezing"}

    def __init__(
        self,
        a: TensorLike,
        theta: TensorLike,
        delta: TensorLike = 1,
        parity: Literal["even", "odd"] = "even",
        include_squeezing: bool = True,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        r"""
        Args:
            a: Magnitude of the displacement

            theta: Angle parameter. A value of :math:`\pi/2` is a good choice.

            delta: Uncertainty in the resulting squeezed state. A value of
                :math:`\Delta = 1` has no squeezing, :math:`\Delta < 1` squeeezes in
                position, and :math:`\Delta > 1` anti-squeezes in position. To achieve
                the desired squeezing, a squeezing gate :math:`S(r)` with
                :math:`r = -\log\Delta` is applied.

            parity: Whether to produce a cat state with even or odd photon number
                parity. Must be one of ``even`` or ``odd``.

            include_squeezing: Whether to include the squeezing operation in the
                decomposition. If ``False``, the input state is assumed to already be
                squeezed.

            wires: The wires this gate acts on

            id: An optional identifier for the gate
        """
        super().__init__(a, theta, delta, wires=wires, id=id)

        self.hyperparameters["parity"] = parity
        self.hyperparameters["include_squeezing"] = include_squeezing

    def error(self) -> AlgorithmicError:
        return GCR(self.data[1], self.data[0], self.data[2], wires=self.wires).error()

    @property
    def resource_params(self):
        return {
            "parity": self.hyperparameters["parity"],
            "has_squeezing": (
                not qml.math.isclose(self.data[2], 1)
                and self.hyperparameters["include_squeezing"]
            ),
        }


def _squeezedcatstate_resources(parity, has_squeezing):
    res = {hqml.XCD: 1, adjoint_resource_rep(GCR): 1, qml.S: 2}

    if parity == "odd":
        res[qml.X] = 1
    if has_squeezing:
        res[hqml.S] = 1

    return res


@qml.register_resources(_squeezedcatstate_resources)
def _squeezedcatstate_decomp(a, theta, delta, wires, **hyperparameters):
    if hyperparameters["parity"] == "odd":
        qml.X(wires[0])

    if not qml.math.isclose(delta, 1) and hyperparameters["include_squeezing"]:
        hqml.Squeezing(-qml.math.log(delta), 0, wires[1])

    hqml.XCD(a, 0, wires)
    qml.adjoint(qml.S)(wires[0])
    qml.adjoint(GCR)(-theta, a, delta, wires)
    qml.S(wires[0])


qml.add_decomps(SqueezedCatState, _squeezedcatstate_decomp)


class GKPState(ErrorOperation, Hybrid):
    r"""GKP-state preparation on a qumode using non-Abelian QSP

    This prepares an approximate GKP state on a qumode using an ancilliary qubit and
    hybrid gates, following the protocol of :footcite:p:`singh2025towards`. The finite-
    energy stabilizers are

    .. math::

        S_{x,\Delta} = e^{i2\sqrt{2}\pi(\cosh \Delta^2\hat{x} - \sinh\Delta^2\hat{p})}
        S_{p,\Delta} = e^{i2\sqrt{2}\pi(\cosh \Delta^2\hat{p} - \sinh\Delta^2\hat{x})}

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: (0,)

    The protocol consists of a series of ``xCD`` and ``yCD`` gates to create the
    superposition of displaced squeezed states. Finally, the disentangling gadget
    :math:`\mathcal{U}(\theta', |\alpha|, \Delta)` is applied to disentangle the qubit
    from the qumode.

    References
    ----------

    .. footbibliography::
    """

    num_wires = 2
    num_params = 1  # pyright: ignore[reportIncompatibleMethodOverride]
    ndim_params = (0,)
    grad_method = None  # pyright: ignore[reportIncompatibleMethodOverride]
    type_signature = (hqml.sa.Qubit(), hqml.sa.Qumode())

    resource_keys = {"logical_state", "has_squeezing", "repetitions"}

    def __init__(
        self,
        delta: TensorLike = 1,
        logical_state: Literal[0, 1] = 0,
        repetitions: int | None = None,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        r"""
        Args:
            delta: Uncertainty in the resulting squeezed state. A value of
                :math:`\Delta = 1` has no squeezing, :math:`\Delta < 1` squeeezes in
                position, and :math:`\Delta > 1` anti-squeezes in position. To achieve
                the desired squeezing, a squeezing gate :math:`S(r)` with
                :math:`r = -\log\Delta` is applied.

            logical_state: Which logical GKP state to prepare. Must be ``0`` or ``1``.

            repetitions: The number of cat state repetitions to use. If None, it will
                infer from the parameter :math:`\Delta`

            wires: The wires this gate acts on

            id: An optional identifier for the gate
        """
        super().__init__(delta, wires=wires, id=id)
        self.hyperparameters["logical_state"] = logical_state

        if repetitions is None:
            repetitions = int(qml.math.floor(0.32 / delta**2).item())

        self.hyperparameters["repetitions"] = repetitions

    def error(self) -> AlgorithmicError:
        pass

    @property
    def resource_params(self):
        return {
            "logical_state": self.hyperparameters["logical_state"],
            "has_squeezing": not qml.math.isclose(self.data[0], 1),
            "repetitions": self.hyperparameters["repetitions"],
        }


def _gkpstate_resources(logical_state, has_squeezing, repetitions):
    resources = {}

    if has_squeezing:
        resources[hqml.S] = 1

    resources[
        qml.resource_rep(SqueezedCatState, has_squeezing=False, parity="even")
    ] = repetitions

    if logical_state == 1:
        resources[hqml.D] = 1

    return resources


@qml.register_resources(_gkpstate_resources)
def _gkpstate_decomp(delta, wires, repetitions=1, logical_state=0):
    a = math.sqrt(math.pi / 2)
    if not qml.math.isclose(delta, 1):
        hqml.Squeezing(-qml.math.log(delta), 0, wires[1])

    for k in range(1, repetitions + 1):
        theta = math.pi / 4 if k < 3 else math.pi / (4 * k)
        SqueezedCatState(a, theta, delta, wires=wires, include_squeezing=False)

    if logical_state == 1:
        hqml.D(a, 0, wires[1])


qml.add_decomps(GKPState, _gkpstate_decomp)


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
