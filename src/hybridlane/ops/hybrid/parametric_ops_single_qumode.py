# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math

import pennylane as qml
from pennylane.decomposition.resources import adjoint_resource_rep
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    pow_rotation,
)
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

from ..mixins import Hybrid
from ..op_math.decompositions.qubit_conditioned_decompositions import (
    decompose_multiqcond_native,
)
from ..qumode import Displacement, Squeezing
from .non_parametric_ops import ConditionalParity


class ConditionalRotation(Operation, Hybrid):
    r"""Qubit-conditioned phase-space rotation :math:`CR(\theta)`

    This operation implements a phase-space rotation on a qumode, conditioned on the
    state of a control qubit. It is given by the unitary expression

    .. math::

        CR(\theta) = \exp[-i \frac{\theta}{2}\sigma_z \hat{n}]

    where :math:`\sigma_z` is the Z operator acting on the qubit, and
    :math:`\hat{n} = \ad a` is the number operator of the qumode (Box III.8 of
    :footcite:p:`liu2026hybrid`). With this definition, the angle parameter ranges
    :math:`\theta \in [0, 4\pi)`.

    References
    ----------

    .. footbibliography::
    """

    num_params = 1
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0,)

    resource_keys = set()

    def __init__(self, theta: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(theta, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def adjoint(self):
        theta = self.parameters[0]
        return ConditionalRotation(-theta, wires=self.wires)

    def pow(self, z: int | float):
        return [ConditionalRotation(self.data[0] * z, self.wires)]

    def simplify(self):
        theta = self.data[0] % (4 * math.pi)

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return ConditionalRotation(theta, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CR", cache=cache
        )


qml.add_decomps("Adjoint(ConditionalRotation)", adjoint_rotation)
qml.add_decomps("Pow(ConditionalRotation)", pow_rotation)
qml.add_decomps("qCond(ConditionalRotation)", decompose_multiqcond_native)

CR = ConditionalRotation
r"""Conditional rotation (CR) gate

.. math::

    CR(\theta) = e^{-i\frac{\theta}{2}\hat{n}Z}

This is an alias for :class:`~hybridlane.ConditionalRotation`
"""


class ConditionalDisplacement(Operation, Hybrid):
    r"""Symmetric conditional displacement gate :math:`CD(\alpha)`

    This is the qubit-conditioned version of the :py:class:`~hybridlane.Displacement`
    gate, given by

    .. math::

        CD(\alpha) = \exp[\sigma_z(\alpha \ad - \alpha^* a)]

    where :math:`\alpha = ae^{i\phi} \in \mathbb{C}` (Box III.7 of
    :footcite:p:`liu2026hybrid`). There also exists a decomposition in terms of
    :py:class:`~hybridlane.ConditionalParity` gates (eq. 20 of
    :footcite:p:`crane2024hybrid`),

    .. math::

        CD(\alpha) = CP~D(i\alpha)~CP^\dagger

    The ``wires`` attribute is assumed to be ``(qubit, qumode)``.

    .. seealso::

        :py:class:`~hybridlane.ops.Displacement`

    References
    ----------

    .. footbibliography::
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self,
        a: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(a, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def pow(self, z: int | float):
        a, phi = self.data
        return [ConditionalDisplacement(a * z, phi, self.wires)]

    def adjoint(self):
        return ConditionalDisplacement(-self.data[0], self.data[1], self.wires)

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)

        if _can_replace(a, 0):
            return qml.Identity(self.wires)

        return ConditionalDisplacement(a, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CD", cache=cache
        )


@qml.register_resources({Displacement: 1, ConditionalParity: 2})
def _cd_parity_decomp(a, phi, wires, **_):
    qml.adjoint(ConditionalParity)(wires)
    Displacement(a, phi + math.pi / 2, wires[1])
    ConditionalParity(wires)


def _cd_to_ecd_resources():
    # Put in function because ECD isn't defined yet
    return {qml.X: 1, EchoedConditionalDisplacement: 1}


@qml.register_resources(_cd_to_ecd_resources)
def _cd_to_ecd(a, phi, wires, **_):
    EchoedConditionalDisplacement(2 * a, phi, wires)
    qml.X(wires[0])


@qml.register_resources({ConditionalDisplacement: 1})
def _pow_cd(a, phi, wires, z, **_):
    ConditionalDisplacement(z * a, phi, wires=wires)


qml.add_decomps(ConditionalDisplacement, _cd_parity_decomp, _cd_to_ecd)
qml.add_decomps("Adjoint(ConditionalDisplacement)", adjoint_rotation)
qml.add_decomps("Pow(ConditionalDisplacement)", _pow_cd)
qml.add_decomps("qCond(ConditionalDisplacement)", decompose_multiqcond_native)

CD = ConditionalDisplacement
r"""Conditional displacement (CD) gate

.. math::

    CD(\alpha) = e^{(\alpha\ad - \alpha^*a)Z}

This is an alias for :class:`~hybridlane.ConditionalDisplacement`
"""


class ConditionalXDisplacement(Operation, Hybrid):
    r"""X-Conditional displacement gate :math:`xCD(\alpha)`

    This is the conditional displacement gate, conditioned on the :math:`X` eigenstates
    of the qubit instead of the usual :math:`Z` eigenstates.

    .. math::

        xCD(\alpha) = H~CD(\alpha)~H
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self,
        a: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(a, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def pow(self, z: int | float):
        a, phi = self.data
        return [ConditionalXDisplacement(a * z, phi, self.wires)]

    def adjoint(self):
        return ConditionalXDisplacement(-self.data[0], self.data[1], self.wires)

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)

        if _can_replace(a, 0):
            return qml.Identity(self.wires)

        return ConditionalXDisplacement(a, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "xCD", cache=cache
        )


@qml.register_resources({ConditionalDisplacement: 1, qml.H: 2})
def _xcd_decomp(a, phi, wires, **_):
    qml.H(wires[0])
    ConditionalDisplacement(a, phi, wires)
    qml.H(wires[0])


@qml.register_resources({ConditionalXDisplacement: 1})
def _pow_xcd(a, phi, wires, z, **_):
    ConditionalXDisplacement(z * a, phi, wires=wires)


qml.add_decomps(ConditionalXDisplacement, _xcd_decomp)
qml.add_decomps("Adjoint(ConditionalXDisplacement)", adjoint_rotation)
qml.add_decomps("Pow(ConditionalXDisplacement)", _pow_xcd)

XCD = ConditionalXDisplacement
r"""X-Conditional displacement (xCD) gate

.. math::

    xCD(\alpha) = e^{(\alpha\ad - \alpha^*a)X}

This is an alias for :class:`~hybridlane.ConditionalXDisplacement`
"""


class ConditionalYDisplacement(Operation, Hybrid):
    r"""Y-Conditional displacement gate :math:`yCD(\alpha)`

    This is the conditional displacement gate, conditioned on the :math:`Y` eigenstates
    of the qubit instead of the usual :math:`Z` eigenstates.

    .. math::

        yCD(\alpha) = S~xCD(\alpha)~S^\dagger
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self,
        a: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(a, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def pow(self, z: int | float):
        a, phi = self.data
        return [ConditionalYDisplacement(a * z, phi, self.wires)]

    def adjoint(self):
        return ConditionalYDisplacement(-self.data[0], self.data[1], self.wires)

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)

        if _can_replace(a, 0):
            return qml.Identity(self.wires)

        return ConditionalYDisplacement(a, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "yCD", cache=cache
        )


def _ycd_resources():
    return {ConditionalXDisplacement: 1, qml.S: 1, adjoint_resource_rep(qml.S): 1}


@qml.register_resources(_ycd_resources)
def _ycd_decomp(a, phi, wires, **_):
    qml.adjoint(qml.S)(wires[0])
    ConditionalXDisplacement(a, phi, wires)
    qml.S(wires[0])


@qml.register_resources({ConditionalYDisplacement: 1})
def _pow_ycd(a, phi, wires, z, **_):
    ConditionalYDisplacement(z * a, phi, wires=wires)


qml.add_decomps(ConditionalYDisplacement, _ycd_decomp)
qml.add_decomps("Adjoint(ConditionalYDisplacement)", adjoint_rotation)
qml.add_decomps("Pow(ConditionalYDisplacement)", _pow_ycd)

YCD = ConditionalYDisplacement
r"""Y-Conditional displacement (yCD) gate

.. math::

    yCD(\alpha) = e^{(\alpha\ad - \alpha^*a)Y}

This is an alias for :class:`~hybridlane.ConditionalYDisplacement`
"""


class ConditionalSqueezing(Operation, Hybrid):
    r"""Qubit-conditioned squeezing gate :math:`CS(\zeta)`

    This gate implements the unitary

    .. math::

        CS(\zeta) = \exp\left[\frac{1}{2}\sigma_z (\zeta^* a^2 - \zeta (\ad)^2)\right]

    where :math:`\zeta = ze^{i\phi} \in \mathbb{C}` (Box IV.3 of
    :footcite:p:`liu2026hybrid`). There exists a decomposition in terms of
    :py:class:`.ConditionalRotation` and :py:class:`~hybridlane.ops.Squeezing` gates

    .. math::

        CS(\zeta) = CR(\pi/2)~S(i\zeta)~CR(-\pi/2)

    .. seealso::

        :class:`~hybridlane.ops.Squeezing`

    References
    ----------

    .. footbibliography::
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self, z: TensorLike, phi: TensorLike, wires: WiresLike, id: str | None = None
    ):
        super().__init__(z, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def pow(self, n: int | float):
        z, phi = self.data
        return [ConditionalSqueezing(z * n, phi, self.wires)]

    def adjoint(self):
        return ConditionalSqueezing(-self.data[0], self.data[1], self.wires)

    def simplify(self):
        z, phi = self.data[0], self.data[1] % (2 * math.pi)

        if _can_replace(z, 0):
            return qml.Identity(self.wires)

        return ConditionalSqueezing(z, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CS", cache=cache
        )


def _cs_decomp_resources():
    return {Squeezing: 1, CR: 1, adjoint_resource_rep(CR): 1}


@qml.register_resources(_cs_decomp_resources)
def _cs_decomp(r, phi, wires, **_):
    qml.adjoint(CR)(math.pi / 2, wires)
    Squeezing(r, phi + math.pi / 2, wires[1])
    CR(math.pi / 2, wires)


@qml.register_resources({ConditionalSqueezing: 1})
def _pow_cs(r, phi, wires, z, **_):
    ConditionalSqueezing(z * r, phi, wires=wires)


qml.add_decomps(ConditionalSqueezing, _cs_decomp)
qml.add_decomps("Adjoint(ConditionalSqueezing)", adjoint_rotation)
qml.add_decomps("Pow(ConditionalSqueezing)", _pow_cs)
qml.add_decomps("qCond(ConditionalSqueezing)", decompose_multiqcond_native)

CS = ConditionalSqueezing
r"""Conditional squeezing (CS) gate

.. math::

    CS(\zeta) = \exp\left[\frac{1}{2}Z (\zeta^* a^2 - \zeta (\ad)^2)\right]

This is an alias for :class:`~hybridlane.ConditionalSqueezing`
"""


class SelectiveQubitRotation(Operation, Hybrid):
    r"""number-Selective Qubit Rotation (SQR) gate :math:`SQR(\theta, \varphi, n)`

    This gate imparts customizeable rotations onto the qubit based on the state
    of the qumode. The unitary expression for this gate is

    .. math::

        SQR(\theta, \varphi) = R_{\varphi}(\theta) \otimes \ket{n}\bra{n}

    with :math:`\theta \in [0, 4\pi)` and :math:`\varphi \in [0, 2\pi)` (Box III.9
    of :footcite:p:`liu2026hybrid`).

    .. note::

        This differs from the vectorized definition in the CVDV paper to act on just a
        single Fock state :math:`\ket{n}`. To match the vectorized version, apply
        multiple SQR gates in series with the appropriate angles and Fock states.

    References
    ----------

    .. footbibliography::
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        n: int,
        wires: WiresLike,
        id: str | None = None,
    ):
        if n < 0:
            raise ValueError(f"Fock state must be >= 0; got {n}")

        # fock state is not trainable
        self.hyperparameters["n"] = n

        super().__init__(theta, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def adjoint(self):
        theta, phi = self.parameters
        n = self.hyperparameters["n"]
        return SelectiveQubitRotation(-theta, phi, n, self.wires)

    def simplify(self):
        theta, phi = self.data
        theta = theta % (4 * math.pi)
        phi = phi % (2 * math.pi)
        n = self.hyperparameters["n"]

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return SelectiveQubitRotation(theta, phi, n, self.wires)

    def pow(self, z: int | float):
        return [
            SelectiveQubitRotation(
                self.data[0] * z, self.data[1], self.hyperparameters["n"], self.wires
            )
        ]

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        hyperparams = dict(metadata[1])
        return cls(data[0], data[1], hyperparams["n"], wires)

    def label(self, decimals=None, base_label=None, cache=None):
        n = self.hyperparameters["n"]
        return super().label(
            decimals=decimals, base_label=base_label or f"SQR_{{{n}}}", cache=cache
        )


SQR = SelectiveQubitRotation
r"""number-Selective Qubit Rotation (SQR) gate`

.. math::

    SQR(\theta, \varphi) = R_{\varphi}(\theta) \otimes \ket{n}\bra{n}

.. seealso::

    This is an alias for :class:`~hybridlane.SelectiveQubitRotation`
"""


@qml.register_resources({SQR: 1})
def _pow_sqr(theta, phi, wires, z, n, **_):
    SQR((theta * z) % (4 * math.pi), phi, n, wires)


qml.add_decomps("Adjoint(SelectiveQubitRotation)", adjoint_rotation)
qml.add_decomps("Pow(SelectiveQubitRotation)", _pow_sqr)


class SelectiveNumberArbitraryPhase(Operation, Hybrid):
    r"""Selective Number-dependent Arbitrary Phase (SNAP) gate :math:`SNAP(\varphi, n)`

    This gate imparts a custom phase onto each Fock state of the qumode. Its expression
    is

    .. math::

        SNAP(\varphi, n) = e^{-i \varphi \sigma_z \ket{n}\bra{n}}

    with :math:`\varphi \in [0, 2\pi)` (Box III.10 of :footcite:p:`liu2026hybrid`). If
    the control qubit starts in the :math:`\ket{0}` state, the :math:`\sigma_z` term
    can be neglected, effectively making this gate purely bosonic. However, because its
    implementation frequently involves an ancilla qubit, it is marked as a hybrid gate.

    .. note::

        This definition differs from the vectorized version presented in the CVDV
        paper, instead applying to a single Fock state. To apply it across multiple
        Fock modes, consider

        .. code:: python

            angles = [0.25, 0.5, 0.75, 1.0]
            fock_states = [0, 3, 7, 10]

            for phi_n, n in zip(angles, fock_states):
                SelectiveNumberArbitraryPhase(phi_n, n, ['q', 'm'])

    References
    ----------

    .. footbibliography::
    """

    num_params = 1
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0,)

    resource_keys = set()

    def __init__(
        self,
        phi: TensorLike,
        n: int,
        wires: WiresLike,
        id: str | None = None,
    ):
        if n < 0:
            raise ValueError(f"Fock state must be >= 0; got {n}")

        self.hyperparameters["n"] = n
        super().__init__(phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def adjoint(self):
        phi = self.parameters[0]
        return SelectiveNumberArbitraryPhase(
            -phi, self.hyperparameters["n"], self.wires
        )

    def pow(self, z: int | float):
        return [
            SelectiveNumberArbitraryPhase(
                self.data[0] * z, self.hyperparameters["n"], self.wires
            )
        ]

    @classmethod
    def _unflatten(cls, data, metadata):
        wires = metadata[0]
        hyperparams = dict(metadata[1])
        return cls(data[0], hyperparams["n"], wires)

    def simplify(self):
        phi = self.data[0] % (2 * math.pi)
        n = self.hyperparameters["n"]

        if _can_replace(phi, 0):
            return qml.Identity(self.wires)

        return SelectiveNumberArbitraryPhase(phi, n, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        n = self.hyperparameters["n"]
        return super().label(
            decimals=decimals, base_label=base_label or f"SNAP_{{{n}}}", cache=cache
        )


SNAP = SelectiveNumberArbitraryPhase
r"""Selective Number-dependent Arbitrary Phase (SNAP) gate

.. math::

    SNAP(\varphi, n) = e^{-i \varphi \sigma_z \ket{n}\bra{n}}

.. seealso::

    This is an alias for :class:`~hybridlane.SelectiveNumberArbitraryPhase`
"""


@qml.register_resources({SQR: 2})
def _snap_to_sqr(phi, wires, n, **_):
    SQR(math.pi, 0, n, wires)
    SQR(-math.pi, phi, n, wires)


qml.add_decomps(SNAP, _snap_to_sqr)
qml.add_decomps("Adjoint(SelectiveNumberArbitraryPhase)", adjoint_rotation)
qml.add_decomps("Pow(SelectiveNumberArbitraryPhase)", pow_rotation)


class JaynesCummings(Operation, Hybrid):
    r"""Jaynes-cummings gate :math:`JC(\theta, \varphi)`, also known as Red-Sideband

    This is the standard interaction for an atom exchanging a photon with a cavity,
    given by the expression

    .. math::

        JC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_- \ad
                                + e^{-i\varphi}\sigma_+ a)]

    where :math:`\sigma_+` (:math:`\sigma_-`) is the raising (lowering) operator of the
    qubit, and :math:`\theta, \varphi \in [0, 2\pi)` (Table III.3 of
    :footcite:p:`liu2026hybrid`).

    .. note::

        We use the convention that the ground state of the qubit (atom)
        :math:`\ket{g} = \ket{0}` and the excited state is :math:`\ket{e} = \ket{1}`.

    .. seealso::

        :py:class:`~hybridlane.AntiJaynesCummings`

    References
    ----------

    .. footbibliography::
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(theta, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def simplify(self):
        theta = self.data[0] % (2 * math.pi)
        phi = self.data[1] % (2 * math.pi)

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return JaynesCummings(theta, phi, self.wires)

    def pow(self, z: int | float):
        return [JaynesCummings(self.data[0] * z, self.data[1], self.wires)]

    def adjoint(self):
        return JaynesCummings(-self.data[0], self.data[1], self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "JC", cache=cache
        )


Red = JaynesCummings
r"""Red sideband gate

.. math::

    JC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_- \ad + e^{-i\varphi}\sigma_+ a)]

.. seealso::

    This is an alias of :class:`~hybridlane.JaynesCummings`
"""

JC = JaynesCummings
r"""Jaynes-Cummings gate

.. math::

    JC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_- \ad
        + e^{-i\varphi}\sigma_+ a)]

.. seealso::

    This is an alias of :class:`~hybridlane.JaynesCummings`
"""


@qml.register_resources({Red: 1})
def _pow_jc(theta, phi, wires, z, **_):
    Red(theta * z, phi, wires)


qml.add_decomps("Adjoint(JaynesCummings)", adjoint_rotation)
qml.add_decomps("Pow(JaynesCummings)", _pow_jc)


class AntiJaynesCummings(Operation, Hybrid):
    r"""Anti-Jaynes-cummings gate :math:`AJC(\theta, \varphi)`, also known as Blue-Sideband

    This is given by the expression (Table III.3 of :footcite:p:`liu2026hybrid`)

    .. math::

        AJC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_
                                + \ad + e^{-i\varphi}\sigma_- a)]

    where :math:`\sigma_+` (:math:`\sigma_-`) is the raising (lowering) operator of the
    qubit, and :math:`\theta, \varphi \in [0, 2\pi)`.

    .. note::

        We use the convention that the ground state of the qubit (atom)
        :math:`\ket{g} = \ket{0}` and the excited state is :math:`\ket{e} = \ket{1}`.

    .. seealso::

        :py:class:`~hybridlane.JaynesCummings`

    References
    ----------

    .. footbibliography::
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(theta, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def simplify(self):
        theta = self.data[0] % (2 * math.pi)
        phi = self.data[1] % (2 * math.pi)

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return AntiJaynesCummings(theta, phi, self.wires)

    def pow(self, z: int | float):
        return [AntiJaynesCummings(self.data[0] * z, self.data[1], self.wires)]

    def adjoint(self):
        return AntiJaynesCummings(-self.data[0], self.data[1], self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "AJC", cache=cache
        )


Blue = AntiJaynesCummings
r"""Blue sideband gate

.. math::

    AJC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_+ \ad + e^{-i\varphi}\sigma_- a)]

.. seealso::

    This is an alias of :class:`~hybridlane.AntiJaynesCummings`
"""

AJC = AntiJaynesCummings
r"""Anti-Jaynes-Cummings gate

.. math::

    AJC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_+ \ad
        + e^{-i\varphi}\sigma_- a)]

.. seealso::

    This is an alias of :class:`~hybridlane.AntiJaynesCummings`
"""


@qml.register_resources({Blue: 1})
def _pow_ajc(theta, phi, wires, z, **_):
    Blue(theta * z, phi, wires)


qml.add_decomps("Adjoint(AntiJaynesCummings)", adjoint_rotation)
qml.add_decomps("Pow(AntiJaynesCummings)", _pow_ajc)


class Rabi(Operation, Hybrid):
    r"""Rabi interaction :math:`RB(\theta)`

    This hybrid gate is given by the expression

    .. math::

        RB(\theta) = \exp[-i\sigma_x (\theta \ad + \theta^*a)]

    where :math:`\theta = re^{i\varphi} \in \mathbb{C}` (Table III.3 of
    :footcite:p:`liu2026hybrid`).

    References
    ----------

    .. footbibliography::
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self, r: TensorLike, phi: TensorLike, wires: WiresLike, id: str | None = None
    ):
        super().__init__(r, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def simplify(self):
        r = self.data[0]
        phi = self.data[1] % (2 * math.pi)

        if _can_replace(r, 0):
            return qml.Identity(self.wires)

        return Rabi(r, phi, self.wires)

    def pow(self, z: int | float):
        return [Rabi(self.data[0] * z, self.data[1], self.wires)]

    def adjoint(self):
        return Rabi(-self.data[0], self.data[1], self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "RB", cache=cache
        )


@qml.register_resources({ConditionalDisplacement: 1, qml.H: 2})
def _rb_to_cd(r, phi, wires, **_):
    qml.H(wires[0])
    ConditionalDisplacement(r, phi - math.pi / 2, wires)
    qml.H(wires[0])


@qml.register_resources({Rabi: 1, qml.H: 2})
def _cd_to_rb(r, phi, wires, **_):
    qml.H(wires[0])
    Rabi(r, phi + math.pi / 2, wires)
    qml.H(wires[0])


@qml.register_resources({Rabi: 1})
def _pow_rb(r, phi, wires, z, **_):
    Rabi(r * z, phi, wires)


qml.add_decomps(Rabi, _rb_to_cd)
qml.add_decomps(ConditionalDisplacement, _cd_to_rb)
qml.add_decomps("Adjoint(Rabi)", adjoint_rotation)
qml.add_decomps("Pow(Rabi)", _pow_rb)


class EchoedConditionalDisplacement(Operation, Hybrid):
    r"""Echoed conditional displacement gate :math:`ECD(\alpha)`

    This is given by the unitary (p. S9 of :footcite:p:`eickbusch2022fast`)

    .. math::

        ECD(\alpha) = X~CD(\alpha/2)

    where :math:`CD(\alpha)` is the :py:class:`~.ConditionalDisplacement` gate.

    References
    ----------

    .. footbibliography::
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    ndim_params = (0, 0)

    resource_keys = set()

    def __init__(
        self,
        a: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        super().__init__(a, phi, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def pow(self, z: int | float):
        a, phi = self.data
        return [EchoedConditionalDisplacement(a * z, phi, self.wires)]

    def adjoint(self):
        return [EchoedConditionalDisplacement(-self.data[0], self.data[1], self.wires)]

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)

        if _can_replace(a, 0):
            return qml.Identity(self.wires)

        return EchoedConditionalDisplacement(a, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "ECD", cache=cache
        )


@qml.register_resources({ConditionalDisplacement: 1, qml.X: 1})
def _ecd_decomp(a, phi, wires, **_):
    ConditionalDisplacement(a / 2, phi, wires=wires)
    qml.X(wires[0])


@qml.register_resources({EchoedConditionalDisplacement: 1})
def _pow_ecd(a, phi, wires, z, **_):
    EchoedConditionalDisplacement(z * a, phi, wires=wires)


qml.add_decomps(EchoedConditionalDisplacement, _ecd_decomp)
qml.add_decomps("Adjoint(EchoedConditionalDisplacement)", adjoint_rotation)
qml.add_decomps("Pow(EchoedConditionalDisplacement)", _pow_ecd)

ECD = EchoedConditionalDisplacement
r"""Echoed-conditional displacement (ECD) gate

.. math::

    ECD(\alpha) = X~CD(\alpha/2)

This is an alias for :class:`~hybridlane.EchoedConditionalDisplacement`
"""


def _can_replace(x, y):
    return (
        not qml.math.is_abstract(x)
        and not qml.math.requires_grad(x)
        and qml.math.allclose(x, y)
    )
