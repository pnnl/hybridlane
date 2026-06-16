# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import math

import numpy as np
import pennylane as qp
from pennylane.decomposition.symbolic_decomposition import (
    adjoint_rotation,
    pow_rotation,
)
from pennylane.operation import CVOperation
from pennylane.ops.cv import _rotation, _two_term_shift_rule
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

import hybridlane as hl

from ..mixins import FockRepresentation
from ..op_math.decompositions.qubit_conditioned_decompositions import to_native_qcond


# Re-export since it matches the convention of Y. Liu
class Displacement(CVOperation, FockRepresentation):
    r"""Phase space displacement gate :math:`D(\alpha)`

    .. math::

       D(\alpha) = \exp[\alpha \ad -\alpha^* a]

    where :math:`\alpha = ae^{i\phi}`.

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0,0)``

    The result of applying a displacement to the vacuum is a coherent state
    :math:`D(\alpha)\ket{0} = \ket{\alpha}`. Displacement gates also compose with a global
    phase (Box IV.1 of :footcite:p:`liu2026hybrid`):

    .. math::

        D(\alpha) D(\beta) = e^{i \mathrm{Im}(\alpha \beta^*)} D(\alpha + \beta).

    Its symplectic representation is given (in standard units) by

    .. math::

        \begin{pmatrix}
            I \\
            \hat{x}' \\
            \hat{p}'
        \end{pmatrix} =
        \begin{pmatrix}
            1 & 0 & 0\\
            \sqrt{2} \mathrm{Re}(\alpha) & 1 & 0 \\
            \sqrt{2} \mathrm{Im}(\alpha) & 0 & 1 \\
        \end{pmatrix}
        \begin{pmatrix}
            I \\
            \hat{x} \\
            \hat{p}
        \end{pmatrix}

    For specific parameter values, it may be obtained like

    >>> D(0.5, 0, wires=0).heisenberg_tr((0,))
    array([[1.    , 0.    , 0.    ],
           [0.7071, 1.    , 0.    ],
           [0.    , 0.    , 1.    ]])

    References
    ----------
    .. footbibliography::
    """

    num_params = 2
    num_wires = 1
    ndim_params = (0, 0)
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / shift
    a = 1
    grad_recipe = (
        [[multiplier, a, shift], [-multiplier, a, -shift]],
        _two_term_shift_rule,
    )

    resource_keys = set()

    def __init__(
        self, a: TensorLike, phi: TensorLike, wires: WiresLike, id: str | None = None
    ):
        super().__init__(a, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        c = math.cos(p[1])
        s = math.sin(p[1])
        scale = math.sqrt(2)
        return hl.math.asarray(
            [
                [1, 0, 0],
                [scale * c * p[0], 1, 0],
                [scale * s * p[0], 0, 1],
            ],
            like=p[0],
        )

    def adjoint(self):
        a, phi = self.parameters
        new_phi = (phi + math.pi) % (2 * math.pi)
        return Displacement(a, new_phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "D", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], a, phi) -> TensorLike:
        ad = hl.math.asarray(hl.Ad.compute_fock_matrix(wire_dims), like=a)
        alpha = a * hl.math.exp(1j * phi)
        op = alpha * ad - hl.math.dag(alpha * ad)
        return hl.math.expm(op)


@qp.register_resources({Displacement: 1})
def _pow_d(a, phi, wires, z, **_):
    Displacement(a * z, phi, wires)


qp.add_decomps("Adjoint(Displacement)", adjoint_rotation)
qp.add_decomps("Pow(Displacement)", _pow_d)
qp.add_decomps("qCond(Displacement)", to_native_qcond(1))

D = Displacement
r"""Phase space displacement gate :math:`D(\alpha)`

.. math::
   D(\alpha) = \exp[\alpha \ad -\alpha^* a]

.. seealso::

    This is an alias of :class:`~hybridlane.Displacement`
"""


# Modify to use -i convention
class Rotation(CVOperation, FockRepresentation):
    r"""Phase space rotation gate :math:`R(\theta)`

    .. math::

        R(\theta) = \exp[-i\theta \hat{n}]

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0,0)``

    The rotation gate is diagonal in the Fock basis:

    >>> R(0.5, wires=0).fock_matrix((3,))
    array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.8776-0.4794j, 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.5403-0.8415j]])

    Its symplectic representation is given (in standard units) by

    .. math::

        \begin{pmatrix}
            I \\
            \hat{x}' \\
            \hat{p}'
        \end{pmatrix} =
        \begin{pmatrix}
            1 & 0 & 0\\
            0 & \cos\theta & \sin\theta \\
            0 & -\sin\theta & \cos\theta \\
        \end{pmatrix}
        \begin{pmatrix}
            I \\
            \hat{x} \\
            \hat{p}
        \end{pmatrix}

    For specific parameter values, it may be obtained like

    >>> R(np.pi/4, wires=0).heisenberg_tr((0,))
    array([[ 1.    ,  0.    ,  0.    ],
           [ 0.    ,  0.7071,  0.7071],
           [ 0.    , -0.7071,  0.7071]])
    """

    num_params = 1
    num_wires = 1
    ndim_params = (0,)
    grad_method = "A"
    grad_recipe = (_two_term_shift_rule,)

    resource_keys = set()

    def __init__(self, theta: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(theta, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        return _rotation(-p[0])

    def adjoint(self):
        return Rotation(-self.parameters[0], wires=self.wires)

    def simplify(self):
        theta = self.data[0]
        if _can_replace(theta, 0):
            return qp.Identity(wires=self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "R", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], theta) -> TensorLike:
        n = hl.math.arange(wire_dims[0], like=theta)
        diag = hl.math.exp(-1j * theta * n)
        return hl.math.diag(diag)


qp.add_decomps("Adjoint(Rotation)", adjoint_rotation)
qp.add_decomps("Pow(Rotation)", pow_rotation)
qp.add_decomps("qCond(Rotation)", to_native_qcond(1))

R = Rotation
r"""Phase space rotation gate :math:`R(\theta)`

.. math::

    R(\theta) = \exp[-i\theta \hat{n}]

.. seealso::

    This is an alias of :class:`~hybridlane.Rotation`
"""


# Re-export since it matches paper convention
class Squeezing(CVOperation, FockRepresentation):
    r"""Phase space squeezing gate :math:`S(\zeta)`

    .. math::
        S(\zeta) = \exp\left[\frac{1}{2}(\zeta^* a^2 - \zeta(\ad)^2)\right].

    where :math:`\zeta = r e^{i\phi}`.

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 2
    * Number of dimensions per parameter: ``(0,0)``
    """

    num_params = 2
    num_wires = 1
    ndim_params = (0, 0)
    grad_method = "A"

    shift = 0.1
    multiplier = 0.5 / math.sinh(shift)
    a = 1
    grad_recipe = (
        [[multiplier, a, shift], [-multiplier, a, -shift]],
        _two_term_shift_rule,
    )

    resource_keys = set()

    def __init__(self, r, phi, wires, id=None):
        super().__init__(r, phi, wires=wires, id=id)

    @staticmethod
    def _heisenberg_rep(p):
        R = _rotation(p[1] / 2)
        return R @ np.diag([1, math.exp(-p[0]), math.exp(p[0])]) @ R.T

    def adjoint(self):
        r, phi = self.parameters
        new_phi = (phi + np.pi) % (2 * np.pi)
        return Squeezing(r, new_phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "S", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], r, phi) -> TensorLike:
        a = hl.math.asarray(hl.A.compute_fock_matrix(wire_dims), like=r)
        ad = hl.math.conj(hl.math.transpose(a))
        zeta = r * hl.math.exp(1j * phi)
        op = 0.5 * (hl.math.conj(zeta) * a @ a - zeta * ad @ ad)
        return hl.math.expm(op)


@qp.register_resources({Squeezing: 1})
def _pow_s(r, phi, wires, z, **_):
    Squeezing(r * z, phi, wires)


qp.add_decomps("Adjoint(Squeezing)", adjoint_rotation)
qp.add_decomps("Pow(Squeezing)", _pow_s)
qp.add_decomps("qCond(Squeezing)", to_native_qcond(1))

S = Squeezing
r"""Phase space squeezing gate :math:`S(\zeta)`

.. math::

    S(\zeta) = \exp\left[\frac{1}{2}(\zeta^* a^2 - \zeta(\ad)^2)\right]

.. seealso::

    This is an alias of :class:`~hybridlane.Squeezing`
"""


# Modify to have -i convention
class Kerr(CVOperation, FockRepresentation):
    r"""Kerr gate :math:`K(\kappa)`

    .. math::

        K(\kappa) = \exp[-i \kappa \hat{n}^2].

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: ``(0,)``

    The Kerr gate is diagonal in the Fock basis:

    >>> K(0.5, wires=0).fock_matrix((3,))
    array([[ 1.    +0.j    ,  0.    +0.j    ,  0.    +0.j    ],
           [ 0.    +0.j    ,  0.8776-0.4794j,  0.    +0.j    ],
           [ 0.    +0.j    ,  0.    +0.j    , -0.4161-0.9093j]])
    """

    num_params = 1
    num_wires = 1
    ndim_params = (0,)
    grad_method = "F"

    resource_keys = set()

    def __init__(self, kappa: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(kappa, wires=wires, id=id)

    def adjoint(self):
        return Kerr(-self.parameters[0], wires=self.wires)

    def simplify(self):
        kappa = self.data[0]
        if _can_replace(kappa, 0):
            return qp.Identity(wires=self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "K", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], kappa) -> TensorLike:
        n = hl.math.arange(wire_dims[0], like=kappa)
        diag = hl.math.exp(-1j * kappa * n**2)
        return hl.math.diag(diag)


qp.add_decomps("Adjoint(Kerr)", adjoint_rotation)
qp.add_decomps("Pow(Kerr)", pow_rotation)

K = Kerr
r"""Kerr gate :math:`K(\kappa)`

.. math::

    K(\kappa) = \exp[-i \kappa \hat{n}^2]

.. seealso::

    This is an alias of :class:`~hybridlane.Kerr`
"""


# Modify for -i convention
class CubicPhase(CVOperation, FockRepresentation):
    r"""Cubic phase shift gate :math:`C(r)`

    .. math::

        C(r) = e^{-i r \hat{x}^3}.

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: ``(0,)``
    """

    num_params = 1
    num_wires = 1
    ndim_params = (0,)
    grad_method = "F"

    resource_keys = set()

    def __init__(self, r: TensorLike, wires: WiresLike, id: str | None = None):
        super().__init__(r, wires=wires, id=id)

    def adjoint(self):
        return CubicPhase(-self.parameters[0], wires=self.wires)

    def simplify(self):
        r = self.data[0]
        if _can_replace(r, 0):
            return qp.Identity(wires=self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "C", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(wire_dims: tuple[int, ...], r) -> TensorLike:
        x = hl.math.asarray(hl.X.compute_fock_matrix(wire_dims), like=r)
        x3 = hl.math.linalg.matrix_power(x, 3)
        return hl.math.expm(-1j * r * x3)


qp.add_decomps("Adjoint(CubicPhase)", adjoint_rotation)
qp.add_decomps("Pow(CubicPhase)", pow_rotation)

C = CubicPhase
r"""Cubic phase shift gate :math:`C(r)`

.. math::

    C(r) = e^{-i r \hat{x}^3}.

.. seealso::

    This is an alias of :class:`~hybridlane.CubicPhase`
"""


class SelectiveNumberArbitraryPhase(CVOperation, FockRepresentation):
    r"""Selective Number-dependent Arbitrary Phase (SNAP) gate :math:`SNAP(\varphi, n)`

    .. math::

        SNAP(\varphi, n) = e^{-i \varphi \ket{n}\bra{n}}

    with :math:`\varphi \in [0, 2\pi)` (Box III.10 of :footcite:p:`liu2026hybrid`).

    **Details**:

    * Number of wires: 1
    * Wire arguments: ``[qumode]``
    * Number of parameters: 1
    * Number of dimensions per parameter: ``(0,)``

    The SNAP gate has a diagonal representation in the Fock basis:

    >>> SNAP(0.5, 2, wires=0).fock_matrix((4,))
    array([[1.    +0.j    , 0.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 1.    +0.j    , 0.    +0.j    , 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.8776+0.4794j, 0.    +0.j    ],
           [0.    +0.j    , 0.    +0.j    , 0.    +0.j    , 1.    +0.j    ]])

    It has a decomposition in terms of SQR gates

    .. math::

        SNAP(\varphi, n) = SQR(-\pi, \varphi, n) SQR(\pi, 0, n).

    Using this decomposition requires dynamic qubit allocation with ``num_work_wires >= 1``:

    .. code-block:: python

        dev = qp.device("default.hybrid", fock_level=8)

        @qp.transforms.decompose(gate_set={hl.SQR}, num_work_wires=1)
        @qp.qnode(dev)
        def circuit():
            hl.SNAP(0.5, 1, wires=0)

    >>> print(qp.draw(circuit)())
    <DynamicWire>: ──Allocate─╭SQR_{1}(3.14,0.00)─╭SQR_{1}(-3.14,0.50)──Deallocate─┤...
                0: ───────────╰SQR_{1}(3.14,0.00)─╰SQR_{1}(-3.14,0.50)─────────────┤...

    For devices that do not support dynamic qubit allocation, this can be resolved at compile
    time with :py:func:`~pennylane.transforms.resolve_dynamic_wires`.

    .. note::

        This definition differs from the vectorized version presented in the CVDV
        paper, instead applying to a single Fock state. To apply it across multiple
        Fock modes, consider

        .. code:: python

            angles = [0.25, 0.5, 0.75, 1.0]
            fock_states = [0, 3, 7, 10]

            for phi_n, n in zip(angles, fock_states):
                SelectiveNumberArbitraryPhase(phi_n, n, 'm')

    References
    ----------

    .. footbibliography::
    """

    num_params = 1
    num_wires = 1
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
            return qp.Identity(self.wires)

        return SelectiveNumberArbitraryPhase(phi, n, self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        n = self.hyperparameters["n"]
        return super().label(
            decimals=decimals, base_label=base_label or f"SNAP_{{{n}}}", cache=cache
        )

    @staticmethod
    def compute_fock_matrix(
        wire_dims: tuple[int, ...], phase, n: int = 0
    ) -> TensorLike:
        indices = hl.math.arange(wire_dims[0], like=phase)
        diag = hl.math.where(
            indices == n,
            hl.math.exp(1j * phase),
            hl.math.ones(wire_dims[0], like=phase) + 0j,
        )
        return hl.math.diag(diag)


SNAP = SelectiveNumberArbitraryPhase
r"""Selective Number-dependent Arbitrary Phase (SNAP) gate

.. math::

    SNAP(\varphi, n) = e^{-i \varphi \ket{n}\bra{n}}

.. seealso::

    This is an alias for :class:`~hybridlane.SelectiveNumberArbitraryPhase`
"""


def _snap_to_sqr_resources():
    return {hl.SQR: 2}


@qp.register_resources(_snap_to_sqr_resources, work_wires={"zeroed": 1})
def _snap_to_sqr(phi, wires, n, **_):
    with qp.allocate(1, "zero", restored=True) as ancilla:
        hl.SQR(math.pi, 0, n, ancilla + wires)
        hl.SQR(-math.pi, phi, n, ancilla + wires)


qp.add_decomps(SNAP, _snap_to_sqr)
qp.add_decomps("Adjoint(SelectiveNumberArbitraryPhase)", adjoint_rotation)
qp.add_decomps("Pow(SelectiveNumberArbitraryPhase)", pow_rotation)


def _can_replace(x, y):
    return (
        not qp.math.is_abstract(x)
        and not qp.math.requires_grad(x)
        and qp.math.allclose(x, y)
    )
