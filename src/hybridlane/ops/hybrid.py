# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math
from typing import Iterable, Optional

import pennylane as qml
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import Wires, WiresLike


def _can_replace(x, y):
    """
    Convenience function that returns true if x is close to y and if
    x does not require grad
    """
    return (
        not qml.math.is_abstract(x)
        and not qml.math.requires_grad(x)
        and qml.math.allclose(x, y)
    )


class Hybrid:
    r"""Mixin for hybrid CV-DV gates

    This mixin adds functionality to split the wires of the gate by type into
    qumodes and qubits. By using this mixin, it enforces the convention that
    qumodes come first in the wire order, followed by qubits.

    This mixin is also used in static analysis passes to type-check circuits.
    """

    num_qumodes: int
    """The number of qumodes the gate acts on"""

    wires: Wires

    def split_wires(self) -> tuple[Wires, Wires]:
        """Splits the wires into qubits and qumodes

        Returns:
            qumodes: The wires representing the qumodes this operator acts on

            qubits: The wires representing the qubits this operator acts on
        """

        if not isinstance(self.wires, Iterable):
            raise ValueError("Expected a hybrid gate acting on at least 2 objects")

        wires = Wires(self.wires)
        qumodes, qubits = wires[: self.num_qumodes], wires[self.num_qumodes :]
        return qumodes, qubits


class ConditionalRotation(Operation, Hybrid):
    r"""Qubit-conditioned phase-space rotation :math:`CR(\theta)`

    This operation implements a phase-space rotation on a qumode, conditioned on the state of a control qubit. It
    is given by the unitary expression

    ..math::

        CR(\theta) = \exp[-i \frac{\theta}{2}\sigma_z \hat{n}]

    where :math:`\sigma_z` is the Z operator acting on the qubit, and :math:`\hat{n} = a^\dagger a`
    is the number operator of the qumode (see Box III.8 of [1]_). With this definition, the angle parameter
    :math:`\theta \in [0, 4\pi)`.

    The ``wires`` attribute is assumed to be ``(qubit, qumode)``.

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 2
    num_qumodes = 1

    def __init__(self, theta: TensorLike, wires: WiresLike, id: Optional[str] = None):
        shape = qml.math.shape(theta)
        if len(shape) > 1:
            raise ValueError(f"Expected scalar parameter; got shape {shape}")

        super().__init__(theta, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

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


class ConditionalParity(Operation, Hybrid):
    r"""Qubit-conditioned number parity gate :math:`CP`

    This gate is a special case of the :py:class:`~.ConditionalRotation` gate, with :math:`CP = CR(\pi)`, resulting
    in the unitary expression

    .. math::

        CP = \exp[-i\frac{\pi}{2}\sigma_z \hat{n}]

    .. seealso::

        :py:class:`~.ConditionalRotation`
    """

    num_wires = 2
    num_qumodes = 1

    def __init__(self, wires: WiresLike, id: Optional[str] = None):
        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(*params, wires, **hyperparameters):
        return [ConditionalRotation(math.pi, wires)]

    def adjoint(self):
        return ConditionalRotation(-math.pi, self.wires)


class SelectiveQubitRotation(Operation, Hybrid):
    r"""number-Selective Qubit Rotation (SQR) gate :math:`SQR(\theta, \varphi, n)`

    This gate imparts customizeable rotations onto the qubit based on the state
    of the qumode. The unitary expression for this gate is

    .. math::

        SQR(\theta, \varphi) = R_{\varphi}(\theta) \otimes \ket{n}\bra{n}

    with :math:`\theta \in [0, 4\pi)` and :math:`\varphi \in [0, 2\pi)` (see Box III.9 of [1]_).

    .. note::

        This differs from the vectorized definition in the CVDV paper to act on just a single Fock state :math:`\ket{n}`. To
        match the vectorized version, apply multiple SQR gates in series with the appropriate angles and Fock states.

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 2
    num_qumodes = 1

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        n: int,
        wires: WiresLike,
        id: Optional[str] = None,
    ):
        theta_shape = qml.math.shape(theta)
        if len(theta_shape) > 1:
            raise ValueError(f"Expected scalar for theta; got shape {theta_shape}")

        phi_shape = qml.math.shape(phi)
        if len(phi_shape) > 1:
            raise ValueError(f"Expected scalar for phi; got shape {phi_shape}")

        if n < 0:
            raise ValueError(f"Fock state must be >= 0; got {n}")

        # fock state is not trainable
        self.hyperparameters["n"] = n

        super().__init__(theta, phi, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @property
    def ndim_params(self):
        return (0, 0)

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


class SelectiveNumberArbitraryPhase(Operation, Hybrid):
    r"""Selective Number-dependent Arbitrary Phase (SNAP) gate :math:`SNAP(\varphi, n)`

    This gate imparts a custom phase onto each Fock state of the qumode. Its expression is

    .. math::

        SNAP(\varphi, n) = e^{-i \varphi \sigma_z \ket{n}\bra{n}}

    with :math:`\varphi \in [0, 2\pi)` (see Box III.10 of [1]_). If the control qubit starts in the :math:`\ket{0}` state, the :math:`\sigma_z` term
    can be neglected, effectively making this gate purely bosonic. However, because its implementation frequently
    involves an ancilla qubit, it is marked as a hybrid gate.

    .. note::

        This definition differs from the vectorized version presented in the CVDV paper, instead applying
        to a single Fock state. To apply it across multiple Fock modes, consider

        .. code:: python

            for phi_n, n in zip(angles, fock_states):
                SelectiveNumberArbitraryPhase(phi_n, n, wires)

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 2
    num_qumodes = 1

    def __init__(
        self,
        phi: TensorLike,
        n: int,
        wires: WiresLike,
        id: Optional[str] = None,
    ):
        shape = qml.math.shape(phi)
        if len(shape) > 1:
            raise ValueError(f"Expected scalar-like for phi; got shape {shape}")

        if n < 0:
            raise ValueError(f"Fock state must be >= 0; got {n}")

        self.hyperparameters["n"] = n
        super().__init__(phi, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

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

    @staticmethod
    def compute_decomposition(*params, wires, **hyperparameters):
        phi = params[0]
        n = hyperparameters["n"]

        # Decomposition in terms of SQR (eq. 235 of [1])
        return [
            SelectiveQubitRotation(math.pi, phi, n, wires),
            SelectiveQubitRotation(-math.pi, phi, n, wires),
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


class JaynesCummings(Operation, Hybrid):
    r"""Jaynes-cummings gate :math:`JC(\theta, \varphi)`, also known as Red-Sideband

    This is the standard interaction for an atom exchanging a photon with a cavity, given by the expression

    .. math::

        JC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_- a^\dagger + e^{-i\varphi}\sigma_+ a)]

    where :math:`\sigma_+` (:math:`\sigma_-`) is the raising (lowering) operator of the qubit, and
    :math:`\theta, \varphi \in [0, 2\pi)`. See Table III.3 of [1]_.

    .. note::

        We use the convention that the ground state of the qubit (atom) :math:`\ket{g} = \ket{0}` and the excited
        state is :math:`\ket{e} = \ket{1}`. This is different from the usual physics definition, but it aligns
        with the quantum information convention that the excited state is :math:`\ket{1}`.

    .. seealso::

        :py:class:`~.AntiJaynesCummings`

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 2
    num_qumodes = 1

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: Optional[str] = None,
    ):
        theta_shape = qml.math.shape(theta)
        if len(theta_shape) > 1:
            raise ValueError(f"Expected scalar for theta; got shape {theta_shape}")

        phi_shape = qml.math.shape(phi)
        if len(phi_shape) > 1:
            raise ValueError(f"Expected scalar for phi; got shape {phi_shape}")

        super().__init__(theta, phi, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @property
    def ndim_params(self):
        return (0, 0)

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


class AntiJaynesCummings(Operation, Hybrid):
    r"""Anti-Jaynes-cummings gate :math:`AJC(\theta, \varphi)`, also known as Blue-Sideband

    This is given by the expression (see Table III.3 of [1]_)

    .. math::

        AJC(\theta, \varphi) = \exp[-i\theta(e^{i\varphi}\sigma_+ a^\dagger + e^{-i\varphi}\sigma_- a)]

    where :math:`\sigma_+` (:math:`\sigma_-`) is the raising (lowering) operator of the qubit, and
    :math:`\theta, \varphi \in [0, 2\pi)`.

    .. note::

        We use the convention that the ground state of the qubit (atom) :math:`\ket{g} = \ket{0}` and the excited
        state is :math:`\ket{e} = \ket{1}`. This is different from the usual physics definition, but it aligns
        with the quantum information convention that the excited state is :math:`\ket{1}`.

    .. seealso::

        :py:class:`~.JaynesCummings`

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 2
    num_qumodes = 1

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: Optional[str] = None,
    ):
        theta_shape = qml.math.shape(theta)
        if len(theta_shape) > 1:
            raise ValueError(f"Expected scalar for theta; got shape {theta_shape}")

        phi_shape = qml.math.shape(phi)
        if len(phi_shape) > 1:
            raise ValueError(f"Expected scalar for phi; got shape {phi_shape}")

        super().__init__(theta, phi, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @property
    def ndim_params(self):
        return (0, 0)

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


class Rabi(Operation, Hybrid):
    r"""Rabi interaction :math:`RB(\theta)`

    This hybrid gate is given by the expression

    .. math::

        RB(\theta) = \exp[-i\theta\sigma_x (a^\dagger + a)]

    where :math:`\theta \in \mathbb{R}` (see Table III.3 of [1]_).

    .. todo::

        Verify if the expression is correct. In Table III.3, it is listed as :math:`\exp[-i\sigma_x (\theta a^\dagger + \theta^*a)]`,
        but since :math:`\theta \in \mathbb{R}`, :math:`\theta^* = \theta`. Hence, I factored it out above.

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 2
    num_qumodes = 1

    def __init__(self, theta: TensorLike, wires: WiresLike, id: Optional[str] = None):
        shape = qml.math.shape(theta)
        if len(shape) > 1:
            raise ValueError(f"Expected scalar for theta; got shape {shape}")

        super().__init__(theta, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

    def simplify(self):
        theta = self.data[0] % (2 * math.pi)

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return Rabi(theta, self.wires)

    def pow(self, z: int | float):
        return [Rabi(self.data[0] * z, self.wires)]

    def adjoint(self):
        return Rabi(-self.data[0], self.wires)


class ConditionalDisplacement(Operation, Hybrid):
    r"""Symmetric conditional displacement gate :math:`CD(\alpha)`

    This is the qubit-conditioned version of the :py:class:`~pennylane.ops.cv.Displacement` gate, given by

    .. math::

        CD(\alpha) = \ket{0}\bra{0} \otimes D(\alpha) + \ket{1}\bra{1} \otimes D(-\alpha)

    where :math:`\alpha = ae^{i\phi} \in \mathbb{C}` (see Box III.7 of [1]_). This can also be written as an exponential

    .. math::

        CD(\alpha) = \exp[\sigma_z(\alpha a^\dagger - \alpha^* a)]

    .. seealso::

        :py:class:`~pennylane.ops.cv.Displacement`

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 2
    num_qumodes = 1

    def __init__(
        self,
        a: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: Optional[str] = None,
    ):
        a_shape = qml.math.shape(a)
        if len(a_shape) > 1:
            raise ValueError(f"Expected scalar for a; got shape {a_shape}")

        phi_shape = qml.math.shape(phi)
        if len(phi_shape) > 1:
            raise ValueError(f"Expected scalar for phi; got shape {phi_shape}")

        super().__init__(a, phi, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @property
    def ndim_params(self):
        return (0, 0)

    def pow(self, z: int | float):
        a = self.data[0] ** z
        phi = self.data[1] * z

        return [ConditionalDisplacement(a, phi, self.wires)]

    def adjoint(self):
        return [ConditionalDisplacement(1 / self.data[0], -self.data[1], self.wires)]

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)

        if _can_replace(a, 0):
            return qml.Identity(self.wires)

        return ConditionalDisplacement(a, phi, self.wires)


class ConditionalBeamsplitter(Operation, Hybrid):
    r"""Qubit-conditioned beamsplitter :math:`CBS(\theta, \varphi)`

    This is a multi-qumode gate conditioned on a qubit. It is given by the expression

    .. math::

        CBS(\theta, \varphi) = \exp[-i\frac{\theta}{2}\sigma_z (e^{i\varphi}a^\dagger b + e^{-i\varphi} ab^\dagger)]

    where :math:`\theta \in [0, 4\pi)` and :math:`\varphi \in [0, \pi)` (see Table III.3 of [1]_).

    .. note::

        Pennylane uses a different convention for the Beamsplitter gate compared to the CVDV paper. In particular,
        one must transform :math:`\theta \rightarrow \theta/2` and :math:`\varphi \rightarrow -(\varphi + \pi/2)` to be compatible
        with the original definition.

    .. seealso::

        :py:class:`~pennylane.ops.cv.Beamsplitter`

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 3
    num_qumodes = 2

    def __init__(
        self,
        theta: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: Optional[str] = None,
    ):
        theta_shape = qml.math.shape(theta)
        if len(theta_shape) > 1:
            raise ValueError(f"Expected scalar for theta; got shape {theta_shape}")

        phi_shape = qml.math.shape(phi)
        if len(phi_shape) > 1:
            raise ValueError(f"Expected scalar for phi; got shape {phi_shape}")

        super().__init__(theta, phi, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @property
    def ndim_params(self):
        return (0, 0)

    def adjoint(self):
        return ConditionalBeamsplitter(-self.data[0], self.data[1], self.wires)

    def pow(self, z: int | float):
        return [ConditionalBeamsplitter(self.data[0] * z, self.data[1], self.wires)]

    def simplify(self):
        theta = self.data[0] % (4 * math.pi)
        phi = self.data[1] % math.pi

        if _can_replace(theta, 0):
            return qml.Identity(self.wires)

        return ConditionalBeamsplitter(theta, phi, self.wires)


class ConditionalTwoModeSqueezing(Operation, Hybrid):
    r"""Qubit-conditioned two-mode squeezing :math:`CTMS(\xi)`

    This is the qubit-conditioned version of the :py:class:`~pennylane.ops.cv.TwoModeSqueezing` gate, given by

    .. math::

        CTMS(\xi) = \exp[\sigma_z (\xi a^\dagger b^\dagger - \xi^* ab)]

    where :math:`\xi = re^{i\phi} \in \mathbb{C}` (see Table III.3 of [1]_).

    .. note::

        This formula differs from the Pennylane implementation by a minus sign (:math:`z \rightarrow -z`).

    .. seealso::

        :py:class:`~pennylane.ops.cv.TwoModeSqueezing`

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 3
    num_qumodes = 2

    def __init__(
        self,
        r: TensorLike,
        phi: TensorLike,
        wires: WiresLike,
        id: Optional[str] = None,
    ):
        r_shape = qml.math.shape(r)
        if len(r_shape) > 1:
            raise ValueError(f"Expected scalar for a; got shape {r_shape}")

        phi_shape = qml.math.shape(phi)
        if len(phi_shape) > 1:
            raise ValueError(f"Expected scalar for phi; got shape {phi_shape}")

        super().__init__(r, phi, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @property
    def ndim_params(self):
        return (0, 0)

    def pow(self, z: int | float):
        a = self.data[0] ** z
        phi = self.data[1] * z

        return [ConditionalTwoModeSqueezing(a, phi, self.wires)]

    def adjoint(self):
        return [
            ConditionalTwoModeSqueezing(1 / self.data[0], -self.data[1], self.wires)
        ]

    def simplify(self):
        a, phi = self.data[0], self.data[1] % (2 * math.pi)

        if _can_replace(a, 0):
            return qml.Identity(self.wires)

        return ConditionalTwoModeSqueezing(a, phi, self.wires)


class ConditionalTwoModeSum(Operation, Hybrid):
    r"""Qubit-conditioned two-mode sum gate :math:`CSUM(\lambda)`

    This is a multi-mode gate conditioned on the state of a qubit, given by the expression

    .. math::

        CSUM(\lambda) = \exp[\frac{\lambda}{2}\sigma_z(a + a^\dagger)(b^\dagger - b)]

    with :math:`\lambda \in \mathbb{R}` (see Table III.3 of [1]_).

    .. seealso::

        :py:class:`~hybridlane.qml.ops.cv.TwoModeSum`

    .. [1] Y. Liu et al, 2024. `arXiv <https://arxiv.org/abs/2407.10381>`_
    """

    num_wires = 3
    num_qumodes = 2

    def __init__(self, lam: TensorLike, wires: WiresLike, id: Optional[str] = None):
        shape = qml.math.shape(lam)
        if len(shape) > 1:
            raise ValueError(f"Expected scalar-like for lambda; got shape {shape}")

        super().__init__(lam, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (0,)

    def adjoint(self):
        lambda_ = self.parameters[0]
        return ConditionalTwoModeSum(-lambda_, wires=self.wires)

    def pow(self, z: int | float):
        return [ConditionalTwoModeSum(self.data[0] * z, self.wires)]

    def simplify(self):
        lambda_ = self.data[0]
        if _can_replace(lambda_, 0):
            return qml.Identity(self.wires)

        return ConditionalTwoModeSum(lambda_, self.wires)


# -----------------------------------
#              Observables
# -----------------------------------

# WordType = Union[
#     PauliWord,
#     BoseWord,
#     QuadX,
#     QuadP,
#     QuadOperator,
#     FockStateProjector,
#     NumberOperator,
#     TensorN,
# ]
# SentenceType = Union[PauliSentence, BoseSentence, PolyXP]

# _cv_words = (
#     BoseWord,
#     QuadX,
#     QuadP,
#     QuadOperator,
#     FockStateProjector,
#     NumberOperator,
#     TensorN,
# )
# _dv_words = (PauliWord,)


# def _is_hermitian(w: WordType | "HybridWord"):
#     if getattr(w, "is_hermitian", False):
#         return True
#     elif isinstance(w, (BoseWord, BoseSentence)):
#         return w == w.adjoint()
#     return False


# # Todo: Do we actually need this?
# We're going to need a hybrid observable that can compute diagonalizing gates and process results for everything
# #  - Diagonalizing gates should be the diagonalizing gates of each subword for a HybridWord
# #  - HybridSentence will require multiple tapes, just like PauliSentence
# class HybridProd(CVObservable, Hybrid):
#     r"""Represents a tensor product of CV and DV observables"""

#     def __init__(self, *ops: Union[Operator, CVObservable]):
#         if not all(op.is_hermitian for op in ops):
#             raise ValueError("All operators must be observables")

#         for op1, op2 in itertools.combinations(ops, 2):
#             if op1.wires.shared_wires(op2):
#                 raise ValueError("All operator terms must be disjoint")

#         self._terms = list(ops)

#     @property
#     def decomposition(self) -> list[Operator]:
#         return self._terms

#     @property
#     def eigval_type(self):
#         # Measurements in phase space yield real numbers
#         float_ops = (QuadX, QuadP, QuadOperator)
#         if any(isinstance(op, float_ops) for op in self._terms):
#             return float

#         # Qubit and fock-space measurements are discrete
#         return int

#     @staticmethod
#     def compute_diagonalizing_gates(
#         *ops: Operator | CVObservable,
#         wires: Wires | Iterable[Hashable] | Hashable,
#         **hyperparams: dict[str, Any],
#     ) -> list[Operator]:
#         diagonalizing_gates: list[Operator] = []

#         for op in ops:
#             if op.has_diagonalizing_gates:
#                 diagonalizing_gates.extend(op.diagonalizing_gates())
#             else:
#                 raise RuntimeError(f"Unable to compute diagonalizing gates for {op}")

#         return diagonalizing_gates

#     @property
#     def pauli_rep(self):
#         reps: list[PauliSentence | None] = [op.pauli_rep for op in self._terms]

#         if all(rep is not None for rep in reps):
#             rep: PauliSentence = reps[0]  # type: ignore
#             if len(reps) >= 2:
#                 for rep2 in reps[1:]:
#                     rep = rep @ rep2

#             return rep

#         return None

#     @property
#     def ev_order(self):
#         ev_orders: list[int | None] = [
#             getattr(op, "ev_order", None) for op in self._terms
#         ]

#         if all(ev_order is not None for ev_order in ev_orders):
#             return max(ev_orders)

#         return None


# class HybridWord(Hashable):
#     r"""Represents a single, atomic hybrid observable.

#     A HybridWord is a tensor product of primitive qubit and qumode
#     observables that can be measured in a single experimental setting.

#     Example: PauliWord({0: 'X'}) @ BoseWord({'a': [(1, 1)]})
#     """

#     def __init__(self, words: Iterable[WordType | "HybridWord"]):
#         self._validate_args(words)

#         subwords = set()
#         for word in words:
#             if isinstance(word, HybridWord):
#                 subwords.update(word.subwords)
#             else:
#                 subwords.add(word)

#         self._subwords: frozenset[WordType] = frozenset(subwords)
#         self._wires = Wires.all_wires([w.wires for w in words])

#     @property
#     def subwords(self) -> frozenset[WordType]:
#         return self._subwords

#     @property
#     def wires(self) -> Wires:
#         return self._wires

#     @property
#     def is_dv(self) -> bool:
#         return any(isinstance(w, _dv_words) for w in self.subwords)

#     @property
#     def is_cv(self) -> bool:
#         return any(isinstance(w, _cv_words) for w in self.subwords)

#     @property
#     def is_hybrid(self) -> bool:
#         return self.is_dv and self.is_cv

#     def operator(self) -> Operator:
#         r"""Returns an operator representing this object"""
#         raise NotImplementedError()

#     def commutator(self, other) -> "HybridWord":
#         r"""Computes the commutator between two objects"""
#         raise NotImplementedError()  # todo: fill this out

#     def _validate_args(self, words: Iterable[WordType | "HybridWord"]):
#         for word in words:
#             if not isinstance(word, (WordType, HybridWord)):
#                 raise ValueError("Cannot construct hybrid word from non-word types")

#             if not _is_hermitian(word):
#                 raise ValueError("Word must be hermitian")

#         # Check for wires being disjoint
#         words = list(words)
#         if len(words) >= 2:
#             wires = Wires.all_wires([w.wires for w in words])
#             expected_wires = sum(len(w.wires) for w in words)

#             if len(wires) != expected_wires:
#                 raise ValueError("Subwords must act on disjoint wires")

#     def __repr__(self):
#         s = " @ ".join(map(repr, self.subwords))
#         return s

#     def __matmul__(self, other):
#         if isinstance(other, WordType):
#             return HybridWord(self, other)
#         elif isinstance(other, SentenceType):
#             raise NotImplementedError()
#         else:
#             raise ValueError(f"Unsupported tensor product with type {type(other)}")

#     def __add__(self, other):
#         if isinstance(other, HybridWord):
#             return HybridSentence({self: 1.0, other: 1.0})

#         raise NotImplementedError()

#     def __mul__(self, scalar):
#         if isinstance(scalar, (int, float)):
#             return HybridSentence({self: scalar})

#         raise NotImplementedError()

#     def __rmul__(self, scalar):
#         """Handles scalar * self."""
#         return self.__mul__(scalar)

#     def __hash__(self):
#         return hash(self._subwords)


# class HybridSentence(Operator):
#     """Represents a linear combination of HybridWord observables."""

#     @property
#     def is_hermitian(self):
#         return True

#     def __init__(self, hword_map: dict[HybridWord, float]):
#         self.hword_map = hword_map

#         # The wires are the union of all wires from all words in the sentence
#         all_wires = qml.wires.Wires.all_wires([hw.wires for hw in hword_map.keys()])
#         super().__init__(wires=all_wires)

#     def operator(self) -> Operator:
#         items = list(self.hword_map.items())
#         ops, coeffs = list(zip(*items))
#         return qml.LinearCombination(coeffs, ops)

#     def __repr__(self):
#         return " + ".join([f"{coeff} * {hw}" for hw, coeff in self.hword_map.items()])

#     def __add__(self, other):
#         new_map = self.hword_map.copy()

#         if isinstance(other, HybridSentence):
#             for hw, coeff in other.hword_map.items():
#                 new_map[hw] = new_map.get(hw, 0) + coeff
#         elif isinstance(other, HybridWord):
#             new_map[other] = new_map.get(other, 0) + 1.0
#         else:
#             raise NotImplementedError()

#         return HybridSentence(new_map)

#     def __mul__(self, scalar):
#         """Handles scaling by a number."""
#         if isinstance(scalar, (int, float)):
#             new_map = {hw: coeff * scalar for hw, coeff in self.hword_map.items()}
#             return HybridSentence(new_map)
#         raise NotImplementedError()

#     def __rmul__(self, scalar):
#         """Handles scalar * self."""
#         return self.__mul__(scalar)

#     def simplify(self, tol=1e-8):
#         """Removes terms with coefficients close to zero."""
#         simplified_map = {
#             hw: coeff for hw, coeff in self.hword_map.items() if abs(coeff) > tol
#         }
#         return HybridSentence(simplified_map)
