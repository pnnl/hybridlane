# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import pennylane as qp
from pennylane.decomposition.resources import adjoint_resource_rep
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    pow_rotation,
)
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

import hybridlane as hl

from ...math.utils import can_replace, concrete_or_error
from ..mixins import FockRepresentation, HybridOperation
from ..op_math.decompositions.qubit_conditioned_decompositions import (
    decompose_multiqcond_native,
)
from ..qumode import Displacement, Squeezing
from .non_parametric_ops import ConditionalParity


class ConditionalRotation(HybridOperation, FockRepresentation):
    r"""Qubit-conditioned phase-space rotation :math:`CR(\theta)`

    .. math::

        CR(\theta) &= \exp[-i \frac{\theta}{2}\sigma_z \hat{n}] \\
                   &= \begin{pmatrix}
                       R(\theta/2) & 0 \\
                       0 & R^\dagger(\theta/2)
                   \end{pmatrix}

    where :math:`\theta \in [0, 4\pi)` (Box III.8 of :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: ``(0,)``

    This is the qubit-conditioned version of the :py:class:`~hybridlane.Rotation` gate.

    >>> hl.qcond(hl.R(0.5, wires=1), control_wires=0)
    ConditionalRotation(1.0, wires=[0, 1])

    Its representation in the Fock basis can be obtained with:

    >>> CR(0.5, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.9689-0.2474j, 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 1.    -0.j    , 0.    -0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    -0.j    , 0.9689+0.2474j]])

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

    def adjoint(self):
        theta = self.parameters[0]
        return ConditionalRotation(-theta, wires=self.wires)  # ty:ignore[unsupported-operator]

    def pow(self, z: int | float):
        return [ConditionalRotation(self.data[0] * z, self.wires)]  # ty:ignore[unsupported-operator]

    def simplify(self):
        theta = self.data[0] % (4 * math.pi)  # ty:ignore[unsupported-operator]

        theta = concrete_or_error(
            None, theta, "Cannot simplify CR when ``theta`` is a tracer"
        )
        if can_replace(theta, 0):
            return qp.Identity(self.wires)

        return ConditionalRotation(theta, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CR", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], theta) -> TensorLike:  # ty:ignore[invalid-method-override]
        r = hl.R.compute_fock_matrix(wire_dims[1:], theta / 2)
        rd = hl.math.conj(hl.math.transpose(r))
        return hl.math.block_diag([r, rd])


qp.add_decomps("Adjoint(ConditionalRotation)", adjoint_rotation)
qp.add_decomps("Pow(ConditionalRotation)", pow_rotation)
qp.add_decomps("qCond(ConditionalRotation)", decompose_multiqcond_native)

CR = ConditionalRotation
r"""Conditional rotation (CR) gate

.. math::

    CR(\theta) = e^{-i\frac{\theta}{2}\hat{n}Z}

This is an alias for :class:`~hybridlane.ConditionalRotation`
"""


class ConditionalDisplacement(HybridOperation, FockRepresentation):
    r"""Symmetric conditional displacement gate :math:`CD(\alpha)`

    .. math::

        CD(\alpha) = \exp[\sigma_z(\alpha \ad - \alpha^* a)]

    where :math:`\alpha = ae^{i\phi} \in \mathbb{C}` (Box III.7 of
    :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    This is the qubit-conditioned version of the :py:class:`~hybridlane.Displacement` gate.

    >>> hl.qcond(hl.D(0.5, 0.0, wires=1), control_wires=0)
    ConditionalDisplacement(0.5, 0.0, wires=[0, 1])

    Its representation in the Fock basis can be obtained with:

    >>> CD(0.5, 0.0, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[ 0.8776+0.j, -0.4794+0.j,  0.    +0.j,  0.    +0.j],
           [ 0.4794+0.j,  0.8776+0.j,  0.    +0.j,  0.    +0.j],
           [ 0.    +0.j,  0.    +0.j,  0.8776-0.j,  0.4794-0.j],
           [ 0.    +0.j,  0.    +0.j, -0.4794-0.j,  0.8776-0.j]])

    There also exists a decomposition in terms of :py:class:`~hybridlane.ConditionalParity`
    gates (eq. 20 of :footcite:p:`crane2024hybrid`),

    .. math::

        CD(\alpha) = CP~D(i\alpha)~CP^\dagger

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

    def pow(self, z: int | float):
        a, phi = self.data
        return [ConditionalDisplacement(a * z, phi, self.wires)]  # ty:ignore[unsupported-operator]

    def adjoint(self):
        return ConditionalDisplacement(-self.data[0], self.data[1], self.wires)  # ty:ignore[unsupported-operator]

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)  # ty:ignore[unsupported-operator]

        a = concrete_or_error(None, a, "Cannot simplify CD when ``a`` is a tracer")
        if can_replace(a, 0):
            return qp.Identity(self.wires)

        return ConditionalDisplacement(a, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CD", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], a, phi) -> TensorLike:  # ty:ignore[invalid-method-override]
        d = hl.D.compute_fock_matrix(wire_dims[1:], a, phi)
        dd = hl.math.conj(hl.math.transpose(d))
        return hl.math.block_diag([d, dd])


@qp.register_resources({Displacement: 1, ConditionalParity: 2})
def _cd_parity_decomp(a, phi, wires, **_):
    qp.adjoint(ConditionalParity)(wires)
    Displacement(a, phi + math.pi / 2, wires[1])
    ConditionalParity(wires)


def _cd_to_ecd_resources():
    # Put in function because ECD isn't defined yet
    return {qp.X: 1, EchoedConditionalDisplacement: 1}


@qp.register_resources(_cd_to_ecd_resources)
def _cd_to_ecd(a, phi, wires, **_):
    EchoedConditionalDisplacement(2 * a, phi, wires)
    qp.X(wires[0])


@qp.register_resources({ConditionalDisplacement: 1})
def _pow_cd(a, phi, wires, z, **_):
    ConditionalDisplacement(z * a, phi, wires=wires)


@qp.register_resources({ConditionalDisplacement: 1})
def _adjoint_cd(a, phi, wires, **_):
    ConditionalDisplacement(a, phi + math.pi, wires=wires)


def _cd_to_xcd_resources():
    return {ConditionalXDisplacement: 1, qp.H: 2}


@qp.register_resources(_cd_to_xcd_resources)
def _cd_to_xcd(a, phi, wires, **_):
    qp.H(wires[0])
    ConditionalXDisplacement(a, phi, wires)
    qp.H(wires[0])


qp.add_decomps(ConditionalDisplacement, _cd_parity_decomp, _cd_to_ecd, _cd_to_xcd)
qp.add_decomps("Adjoint(ConditionalDisplacement)", _adjoint_cd)
qp.add_decomps("Pow(ConditionalDisplacement)", _pow_cd)
qp.add_decomps("qCond(ConditionalDisplacement)", decompose_multiqcond_native)

CD = ConditionalDisplacement
r"""Conditional displacement (CD) gate

.. math::

    CD(\alpha) = e^{(\alpha\ad - \alpha^*a)Z}

This is an alias for :class:`~hybridlane.ConditionalDisplacement`
"""


class ConditionalXDisplacement(HybridOperation, FockRepresentation):
    r"""X-Conditional displacement gate :math:`xCD(\alpha)`

    .. math::

            xCD(\alpha) = \exp[\sigma_x(\alpha \ad - \alpha^* a)]

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    This is the conditional displacement gate, conditioned on the :math:`X` eigenstates
    of the qubit instead of the usual :math:`Z` eigenstates.

    .. math::

        xCD(\alpha) = H~CD(\alpha)~H

    Its representation in the Fock basis can be obtained with:

    .. skip: next

    >>> XCD(0.5, 0.0, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[ 0.8776+0.j,  0.    +0.j,  0.    +0.j, -0.4794+0.j],
           [ 0.    +0.j,  0.8776+0.j,  0.4794+0.j,  0.    +0.j],
           [ 0.    +0.j, -0.4794+0.j,  0.8776+0.j,  0.    +0.j],
           [ 0.4794+0.j,  0.    +0.j,  0.    +0.j,  0.8776+0.j]])
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

    def pow(self, z: int | float):
        a, phi = self.data
        return [ConditionalXDisplacement(a * z, phi, self.wires)]  # ty:ignore[unsupported-operator]

    def adjoint(self):
        return ConditionalXDisplacement(-self.data[0], self.data[1], self.wires)  # ty:ignore[unsupported-operator]

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)  # ty:ignore[unsupported-operator]

        a = concrete_or_error(None, a, "Cannot simplify xCD when ``a`` is a tracer")
        if can_replace(a, 0):
            return qp.Identity(self.wires)

        return ConditionalXDisplacement(a, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "xCD", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], a, phi) -> TensorLike:  # ty:ignore[invalid-method-override]
        cd = CD.compute_fock_matrix(wire_dims, a, phi)
        dims = {i: dim for i, dim in enumerate(wire_dims)}
        h = hl.math.expand_matrix(
            qp.H.compute_matrix(), (0,), wire_dims=dims, wire_order=(0, 1)
        )
        return h @ cd @ h  # ty:ignore[unsupported-operator]


@qp.register_resources({ConditionalDisplacement: 1, qp.H: 2})
def _xcd_decomp(a, phi, wires, **_):
    qp.H(wires[0])
    ConditionalDisplacement(a, phi, wires)
    qp.H(wires[0])


@qp.register_resources({ConditionalXDisplacement: 1})
def _adjoint_xcd(a, phi, wires, **_):
    ConditionalXDisplacement(a, phi + math.pi, wires=wires)


@qp.register_resources({ConditionalXDisplacement: 1})
def _pow_xcd(a, phi, wires, z, **_):
    ConditionalXDisplacement(z * a, phi, wires=wires)


qp.add_decomps(ConditionalXDisplacement, _xcd_decomp)
qp.add_decomps("Adjoint(ConditionalXDisplacement)", _adjoint_xcd)
qp.add_decomps("Pow(ConditionalXDisplacement)", _pow_xcd)

XCD = ConditionalXDisplacement
r"""X-Conditional displacement (xCD) gate

.. math::

    xCD(\alpha) = e^{(\alpha\ad - \alpha^*a)X}

This is an alias for :class:`~hybridlane.ConditionalXDisplacement`
"""


class ConditionalYDisplacement(HybridOperation, FockRepresentation):
    r"""Y-Conditional displacement gate :math:`yCD(\alpha)`

    .. math::

            yCD(\alpha) = \exp[\sigma_y(\alpha \ad - \alpha^* a)]

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    This is the conditional displacement gate, conditioned on the :math:`Y` eigenstates
    of the qubit instead of the usual :math:`Z` eigenstates.

    .. math::

        yCD(\alpha) = S~xCD(\alpha)~S^\dagger

    Its representation in the Fock basis can be obtained with:

    .. skip: next

    >>> YCD(0.5, 0.0, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[0.8776+0.j    , 0.    +0.j    , 0.    -0.j    , 0.    +0.4794j],
           [0.    +0.j    , 0.8776+0.j    , 0.    -0.4794j, 0.    -0.j    ],
           [0.    +0.j    , 0.    -0.4794j, 0.8776+0.j    , 0.    +0.j    ],
           [0.    +0.4794j, 0.    +0.j    , 0.    +0.j    , 0.8776+0.j    ]])
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

    def pow(self, z: int | float):
        a, phi = self.data
        return [ConditionalYDisplacement(a * z, phi, self.wires)]  # ty:ignore[unsupported-operator]

    def adjoint(self):
        return ConditionalYDisplacement(-self.data[0], self.data[1], self.wires)  # ty:ignore[unsupported-operator]

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)  # ty:ignore[unsupported-operator]

        a = concrete_or_error(None, a, "Cannot simplify yCD when ``a`` is a tracer")
        if can_replace(a, 0):
            return qp.Identity(self.wires)

        return ConditionalYDisplacement(a, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "yCD", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], a, phi) -> TensorLike:  # ty:ignore[invalid-method-override]
        xcd = XCD.compute_fock_matrix(wire_dims, a, phi)
        dims = {i: dim for i, dim in enumerate(wire_dims)}
        s = hl.math.expand_matrix(
            qp.S.compute_matrix(), (0,), wire_dims=dims, wire_order=(0, 1)
        )
        return s @ xcd @ hl.math.conj(hl.math.transpose(s))  # ty:ignore[unsupported-operator]


def _ycd_resources():
    return {ConditionalXDisplacement: 1, qp.S: 1, adjoint_resource_rep(qp.S): 1}


@qp.register_resources(_ycd_resources)
def _ycd_decomp(a, phi, wires, **_):
    qp.adjoint(qp.S)(wires[0])
    ConditionalXDisplacement(a, phi, wires)
    qp.S(wires[0])


@qp.register_resources({ConditionalYDisplacement: 1})
def _adjoint_ycd(a, phi, wires, **_):
    ConditionalYDisplacement(a, phi + math.pi, wires=wires)


@qp.register_resources({ConditionalYDisplacement: 1})
def _pow_ycd(a, phi, wires, z, **_):
    ConditionalYDisplacement(z * a, phi, wires=wires)


qp.add_decomps(ConditionalYDisplacement, _ycd_decomp)
qp.add_decomps("Adjoint(ConditionalYDisplacement)", _adjoint_ycd)
qp.add_decomps("Pow(ConditionalYDisplacement)", _pow_ycd)

YCD = ConditionalYDisplacement
r"""Y-Conditional displacement (yCD) gate

.. math::

    yCD(\alpha) = e^{(\alpha\ad - \alpha^*a)Y}

This is an alias for :class:`~hybridlane.ConditionalYDisplacement`
"""


class ConditionalSqueezing(HybridOperation, FockRepresentation):
    r"""Qubit-conditioned squeezing gate :math:`CS(\zeta)`

    .. math::

        CS(\zeta) &= \exp\left[\frac{1}{2}\sigma_z (\zeta^* a^2 - \zeta (\ad)^2)\right] \\
                    &= \begin{pmatrix}
                        S(\zeta) & 0 \\
                        0 & S^\dagger(\zeta)
                    \end{pmatrix}

    where :math:`\zeta = ze^{i2\theta} \in \mathbb{C}` (Box IV.3 of :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    This is the qubit-conditioned version of the :py:class:`~hybridlane.Squeezing` gate.

    >>> hl.qcond(hl.S(0.5, 0.0, wires=1), control_wires=0)
    ConditionalSqueezing(0.5, 0.0, wires=[0, 1])

    Its representation in the Fock basis can be obtained with:

    .. skip: next

    >>> CS(0.5, 0.0, wires=(0, 1)).fock_matrix({0: 2, 1: 3})
    array([[ 0.9381+0.j,  0.    +0.j,  0.3462+0.j,  0.    +0.j,  0.    +0.j,
             0.    +0.j],
           [ 0.    +0.j,  1.    +0.j,  0.    +0.j,  0.    +0.j,  0.    +0.j,
             0.    +0.j],
           [-0.3462+0.j,  0.    +0.j,  0.9381+0.j,  0.    +0.j,  0.    +0.j,
             0.    +0.j],
           [ 0.    +0.j,  0.    +0.j,  0.    +0.j,  0.9381-0.j,  0.    -0.j,
            -0.3462-0.j],
           [ 0.    +0.j,  0.    +0.j,  0.    +0.j,  0.    -0.j,  1.    -0.j,
             0.    -0.j],
           [ 0.    +0.j,  0.    +0.j,  0.    +0.j,  0.3462-0.j,  0.    -0.j,
             0.9381-0.j]])

    There exists a decomposition in terms of :py:class:`.ConditionalRotation` and
    :py:class:`~hybridlane.ops.Squeezing` gates

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

    def pow(self, n: int | float):  # ty:ignore[invalid-method-override]
        z, theta = self.data
        return [ConditionalSqueezing(z * n, theta, self.wires)]  # ty:ignore[unsupported-operator]

    def adjoint(self):
        return ConditionalSqueezing(-self.data[0], self.data[1], self.wires)  # ty:ignore[unsupported-operator]

    def simplify(self):
        z, theta = self.data[0], self.data[1] % math.pi  # ty:ignore[unsupported-operator]

        z = concrete_or_error(None, z, "Cannot simplify CS when ``z`` is a tracer")
        if can_replace(z, 0):
            return qp.Identity(self.wires)

        return ConditionalSqueezing(z, theta, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "CS", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], z, theta) -> TensorLike:  # ty:ignore[invalid-method-override]
        s = hl.S.compute_fock_matrix(wire_dims[1:], z, theta)
        sd = hl.math.dag(s)
        return hl.math.block_diag([s, sd])


def _cs_decomp_resources():
    return {Squeezing: 1, CR: 1, adjoint_resource_rep(CR): 1}


@qp.register_resources(_cs_decomp_resources)
def _cs_decomp(r, theta, wires, **_):
    qp.adjoint(CR)(math.pi / 2, wires)
    Squeezing(r, theta + math.pi / 4, wires[1])
    CR(math.pi / 2, wires)


@qp.register_resources({ConditionalSqueezing: 1})
def _pow_cs(r, phi, wires, z, **_):
    ConditionalSqueezing(z * r, phi, wires=wires)


qp.add_decomps(ConditionalSqueezing, _cs_decomp)
qp.add_decomps("Adjoint(ConditionalSqueezing)", adjoint_rotation)
qp.add_decomps("Pow(ConditionalSqueezing)", _pow_cs)
qp.add_decomps("qCond(ConditionalSqueezing)", decompose_multiqcond_native)

CS = ConditionalSqueezing
r"""Conditional squeezing (CS) gate

.. math::

    CS(\zeta) = \exp\left[\frac{1}{2}Z (\zeta^* a^2 - \zeta (\ad)^2)\right]

This is an alias for :class:`~hybridlane.ConditionalSqueezing`
"""


class SelectiveQubitRotation(HybridOperation, FockRepresentation):
    r"""number-Selective Qubit Rotation (SQR) gate :math:`SQR(\theta, \varphi, n)`

    .. math::

        SQR(\theta, \varphi) = R_{\varphi}(\theta) \otimes \ket{n}\bra{n}

    with :math:`\theta \in [0, 4\pi)` and :math:`\varphi \in [0, 2\pi)` (Box III.9
    of :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    Its representation in the Fock basis can be obtained with:

    >>> SQR(0.5, 0.0, 1, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.9689+0.j    , 0.    +0.j    , 0.    -0.2474j],
           [0.    +0.j    , 0.    +0.j    , 1.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    -0.2474j, 0.    +0.j    , 0.9689+0.j    ]])

    .. note::

        This differs from the vectorized definition in :footcite:p:`liu2026hybrid`
        to act on just a single Fock state :math:`\ket{n}`. To match the vectorized version,
        apply multiple SQR gates in series with the appropriate angles and Fock states.

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

    def adjoint(self):
        theta, phi = self.parameters
        n = self.hyperparameters["n"]
        return SelectiveQubitRotation(-theta, phi, n, self.wires)  # ty:ignore[unsupported-operator]

    def simplify(self):
        theta, phi = self.data
        theta = theta % (4 * math.pi)  # ty:ignore[unsupported-operator]
        phi = phi % (2 * math.pi)  # ty:ignore[unsupported-operator]
        n = self.hyperparameters["n"]

        theta = concrete_or_error(
            None, theta, "Cannot simplify SQR when ``theta`` is a tracer"
        )
        if can_replace(theta, 0):
            return qp.Identity(self.wires)

        return SelectiveQubitRotation(theta, phi, n, self.wires)

    def pow(self, z: int | float):
        return [
            SelectiveQubitRotation(
                self.data[0] * z, self.data[1], self.hyperparameters["n"], self.wires  # ty:ignore[unsupported-operator]
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

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], theta, phi, n) -> TensorLike:  # ty:ignore[invalid-method-override]
        def r(theta, phi):
            sigma = (
                hl.math.cos(phi) * qp.X.compute_matrix()
                + hl.math.sin(phi) * qp.Y.compute_matrix()
            )
            return hl.math.expm(-1j * theta / 2 * sigma)

        blocks = [hl.math.eye(2)] * wire_dims[1]
        blocks[n] = r(theta, phi)
        mat = hl.math.block_diag(blocks)
        mat = hl.math.expand_matrix(
            mat, (1, 0), wire_dims={0: wire_dims[0], 1: wire_dims[1]}, wire_order=(0, 1)
        )
        return mat


SQR = SelectiveQubitRotation
r"""number-Selective Qubit Rotation (SQR) gate`

.. math::

    SQR(\theta, \varphi) = R_{\varphi}(\theta) \otimes \ket{n}\bra{n}

.. seealso::

    This is an alias for :class:`~hybridlane.SelectiveQubitRotation`
"""


@qp.register_resources({SQR: 1})
def _pow_sqr(theta, phi, wires, z, n, **_):
    SQR((theta * z) % (4 * math.pi), phi, n, wires)


qp.add_decomps("Adjoint(SelectiveQubitRotation)", adjoint_rotation)
qp.add_decomps("Pow(SelectiveQubitRotation)", _pow_sqr)


class JaynesCummings(HybridOperation, FockRepresentation):
    r"""Jaynes-cummings gate :math:`JC(\theta, \varphi)`, also known as Red-Sideband

    .. math::

        JC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_- \ad
                                + e^{-i\varphi}\sigma_+ a)]

    where :math:`\sigma_+` (:math:`\sigma_-`) is the raising (lowering) operator of the
    qubit, and :math:`\theta, \varphi \in [0, 2\pi)` (Table III.3 of
    :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    Its representation in the Fock basis can be obtained with:

    >>> JC(0.5, 0.0, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.8776+0.j    , 0.    -0.4794j, 0.    +0.j    ],
           [0.    +0.j    , 0.    -0.4794j, 0.8776+0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 1.    +0.j    ]])

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

    def simplify(self):
        theta = self.data[0]
        phi = self.data[1] % (2 * math.pi)  # ty:ignore[unsupported-operator]

        theta = concrete_or_error(
            None, theta, "Cannot simplify JC when ``theta`` is a tracer"
        )
        if can_replace(theta, 0):
            return qp.Identity(self.wires)

        return JaynesCummings(theta, phi, self.wires)

    def pow(self, z: int | float):
        return [JaynesCummings(self.data[0] * z, self.data[1], self.wires)]  # ty:ignore[unsupported-operator]

    def adjoint(self):
        return JaynesCummings(-self.data[0], self.data[1], self.wires)  # ty:ignore[unsupported-operator]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "JC", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], theta, phi) -> TensorLike:  # ty:ignore[invalid-method-override]
        ad = hl.Ad.compute_fock_matrix(wire_dims[1:])
        ad = hl.math.asarray(ad, like=theta)
        sigma_minus = hl.math.asarray([[0, 1], [0, 0]], like=theta)
        term = hl.math.exp(1j * phi) * hl.math.kron(sigma_minus, ad)
        return hl.math.expm(-1j * theta * (term + hl.math.dag(term)))


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


@qp.register_resources({Red: 1})
def _pow_jc(theta, phi, wires, z, **_):
    Red(theta * z, phi, wires)


qp.add_decomps("Adjoint(JaynesCummings)", adjoint_rotation)
qp.add_decomps("Pow(JaynesCummings)", _pow_jc)


class AntiJaynesCummings(HybridOperation, FockRepresentation):
    r"""Anti-Jaynes-cummings gate :math:`AJC(\theta, \varphi)`, also known as Blue-Sideband

    .. math::

        AJC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_
                                + \ad + e^{-i\varphi}\sigma_- a)]

    where :math:`\sigma_+` (:math:`\sigma_-`) is the raising (lowering) operator of the
    qubit, and :math:`\theta, \varphi \in [0, 2\pi)` (Table III.3 of
    :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    Its representation in the Fock basis can be obtained with:

    >>> AJC(0.5, 0.0, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[0.8776+0.j    , 0.    +0.j    , 0.    +0.j    , 0.    -0.4794j],
           [0.    +0.j    , 1.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 1.    +0.j    , 0.    +0.j    ],
           [0.    -0.4794j, 0.    +0.j    , 0.    +0.j    , 0.8776+0.j    ]])

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

    def simplify(self):
        theta = self.data[0]
        phi = self.data[1] % (2 * math.pi)  # ty:ignore[unsupported-operator]

        theta = concrete_or_error(
            None, theta, "Cannot simplify AJC when ``theta`` is a tracer"
        )
        if can_replace(theta, 0):
            return qp.Identity(self.wires)

        return AntiJaynesCummings(theta, phi, self.wires)

    def pow(self, z: int | float):
        return [AntiJaynesCummings(self.data[0] * z, self.data[1], self.wires)]  # ty:ignore[unsupported-operator]

    def adjoint(self):
        return AntiJaynesCummings(-self.data[0], self.data[1], self.wires)  # ty:ignore[unsupported-operator]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "AJC", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], theta, phi) -> TensorLike:  # ty:ignore[invalid-method-override]
        ad = hl.Ad.compute_fock_matrix(wire_dims[1:])
        ad = hl.math.asarray(ad, like=theta)
        sigma_plus = hl.math.asarray([[0, 0], [1, 0]], like=theta)
        term = hl.math.exp(1j * phi) * hl.math.kron(sigma_plus, ad)
        return hl.math.expm(-1j * theta * (term + hl.math.dag(term)))


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


@qp.register_resources({Blue: 1})
def _pow_ajc(theta, phi, wires, z, **_):
    Blue(theta * z, phi, wires)


qp.add_decomps("Adjoint(AntiJaynesCummings)", adjoint_rotation)
qp.add_decomps("Pow(AntiJaynesCummings)", _pow_ajc)


class Rabi(HybridOperation, FockRepresentation):
    r"""Rabi interaction :math:`RB(\theta)`

    .. math::

        RB(\theta) = \exp[-i\sigma_x (\theta \ad + \theta^*a)]

    where :math:`\theta = re^{i\varphi} \in \mathbb{C}` (Table III.3 of
    :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    Its representation in the Fock basis can be obtained with:

    >>> Rabi(0.5, 0.0, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[0.8776+0.j    , 0.    +0.j    , 0.    +0.j    , 0.    -0.4794j],
           [0.    +0.j    , 0.8776+0.j    , 0.    -0.4794j, 0.    +0.j    ],
           [0.    +0.j    , 0.    -0.4794j, 0.8776+0.j    , 0.    +0.j    ],
           [0.    -0.4794j, 0.    +0.j    , 0.    +0.j    , 0.8776+0.j    ]])

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

    def simplify(self):
        r = self.data[0]
        phi = self.data[1] % (2 * math.pi)  # ty:ignore[unsupported-operator]

        r = concrete_or_error(None, r, "Cannot simplify RB when ``r`` is a tracer")
        if can_replace(r, 0):
            return qp.Identity(self.wires)

        return Rabi(r, phi, self.wires)

    def pow(self, z: int | float):
        return [Rabi(self.data[0] * z, self.data[1], self.wires)]  # ty:ignore[unsupported-operator]

    def adjoint(self):
        return Rabi(-self.data[0], self.data[1], self.wires)  # ty:ignore[unsupported-operator]

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "RB", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], r, phi) -> TensorLike:  # ty:ignore[invalid-method-override]
        ad = hl.Ad.compute_fock_matrix(wire_dims[1:])
        ad = hl.math.asarray(ad, like=r)
        x = qp.X.compute_matrix()
        x = hl.math.asarray(x, like=r)
        term = hl.math.exp(1j * phi) * hl.math.kron(x, ad)
        return hl.math.expm(-1j * r * (term + hl.math.dag(term)))


@qp.register_resources({ConditionalDisplacement: 1, qp.H: 2})
def _rb_to_cd(r, phi, wires, **_):
    qp.H(wires[0])
    ConditionalDisplacement(r, phi - math.pi / 2, wires)
    qp.H(wires[0])


@qp.register_resources({Rabi: 1, qp.H: 2})
def _cd_to_rb(r, phi, wires, **_):
    qp.H(wires[0])
    Rabi(r, phi + math.pi / 2, wires)
    qp.H(wires[0])


@qp.register_resources({Rabi: 1})
def _pow_rb(r, phi, wires, z, **_):
    Rabi(r * z, phi, wires)


qp.add_decomps(Rabi, _rb_to_cd)
qp.add_decomps(ConditionalDisplacement, _cd_to_rb)
qp.add_decomps("Adjoint(Rabi)", adjoint_rotation)
qp.add_decomps("Pow(Rabi)", _pow_rb)


class EchoedConditionalDisplacement(HybridOperation, FockRepresentation):
    r"""Echoed conditional displacement gate :math:`ECD(\alpha)`

    .. math::

        ECD(\alpha) = X~CD(\alpha/2)

    where :math:`CD(\alpha)` is the :py:class:`~.ConditionalDisplacement` gate
    (p. S9 of :footcite:p:`eickbusch2022fast`).

    **Details**:

    * Number of wires: 2
    * Wire arguments: ``[qubit, qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0, 0)``

    Its representation in the Fock basis can be obtained with:

    >>> ECD(0.5, 0.0, wires=(0, 1)).fock_matrix({0: 2, 1: 2})
    array([[ 0.    +0.j,  0.    +0.j,  0.9689+0.j,  0.2474+0.j],
           [ 0.    +0.j,  0.    +0.j, -0.2474+0.j,  0.9689+0.j],
           [ 0.9689+0.j, -0.2474+0.j,  0.    +0.j,  0.    +0.j],
           [ 0.2474+0.j,  0.9689+0.j,  0.    +0.j,  0.    +0.j]])

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

    def pow(self, z: int | float):
        a, phi = self.data
        return [EchoedConditionalDisplacement(a * z, phi, self.wires)]  # ty:ignore[unsupported-operator]

    def adjoint(self):
        return [EchoedConditionalDisplacement(-self.data[0], self.data[1], self.wires)]  # ty:ignore[unsupported-operator]

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)  # ty:ignore[unsupported-operator]

        a = concrete_or_error(None, a, "Cannot simplify ECD when ``a`` is a tracer")
        if can_replace(a, 0):
            return qp.Identity(self.wires)

        return EchoedConditionalDisplacement(a, phi, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "ECD", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], a, phi) -> TensorLike:  # ty:ignore[invalid-method-override]
        dims = {i: d for i, d in enumerate(wire_dims)}
        cd = CD.compute_fock_matrix(wire_dims, a / 2, phi)
        x = qp.X.compute_matrix()
        x = hl.math.asarray(x, like=a)
        x = hl.math.expand_matrix(x, (0,), wire_dims=dims, wire_order=(0, 1))
        return x @ cd


@qp.register_resources({ConditionalDisplacement: 1, qp.X: 1})
def _ecd_decomp(a, phi, wires, **_):
    ConditionalDisplacement(a / 2, phi, wires=wires)
    qp.X(wires[0])


@qp.register_resources({EchoedConditionalDisplacement: 1})
def _pow_ecd(a, phi, wires, z, **_):
    EchoedConditionalDisplacement(z * a, phi, wires=wires)


qp.add_decomps(EchoedConditionalDisplacement, _ecd_decomp)
qp.add_decomps("Adjoint(EchoedConditionalDisplacement)", adjoint_rotation)
qp.add_decomps("Pow(EchoedConditionalDisplacement)", _pow_ecd)

ECD = EchoedConditionalDisplacement
r"""Echoed-conditional displacement (ECD) gate

.. math::

    ECD(\alpha) = X~CD(\alpha/2)

This is an alias for :class:`~hybridlane.EchoedConditionalDisplacement`
"""
