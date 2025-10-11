# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

r"""Module containing the native bosonic gates of the ion trap"""

import math

import pennylane as qml
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

import hybridlane as hqml

from ...ops.hybrid.parametric_ops_single_qumode import _can_replace
from ...ops.mixins import Hybrid

Red = hqml.Red
Blue = hqml.Blue


class ConditionalXDisplacement(Operation, Hybrid):
    r"""Symmetric conditional displacement gate :math:`C_xD(\alpha)`

    This is the qubit-conditioned version of the :py:class:`~hybridlane.ops.Displacement` gate, given by

    .. math::

        CD(\beta) &= \exp[\sigma_x(\beta \ad - \beta^* a)]

    which differs from :class:`~hybridlane.ops.ConditionalDisplacement` due to the :math:`\sigma_x` factor
    instead of :math:`\sigma_z`.

    This is represented by the hardware instruction ``SDF``, and it can only be used on hardware qumode ``a0m1``
    """

    num_params = 2
    num_wires = 2
    num_qumodes = 1
    grad_method = "F"

    resource_keys = set()

    def __init__(
        self,
        beta_re: TensorLike,
        beta_im: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        r"""
        Args:
            beta_re: The real part of :math:`\beta`

            beta_im: The imaginary part of :math:`\beta`

            wires: The wires this gate acts on, in the format ``(qubit, qumode)``
        """
        super().__init__(beta_re, beta_im, wires=wires, id=id)

    def pow(self, n: int | float):
        beta_re, beta_im = self.data
        return [ConditionalXDisplacement(beta_re * n, beta_im * n, self.wires)]

    def adjoint(self):
        return ConditionalXDisplacement(-self.data[0], -self.data[1], self.wires)

    def simplify(self):
        beta_re, beta_im = self.data

        if _can_replace(beta_re, 0) and _can_replace(beta_im, 0):
            return qml.Identity(self.wires)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "SDF", cache=cache
        )

    @property
    def resource_params(self):
        return {}


class ConditionalXSqueezing(Operation, Hybrid):
    r"""Qubit-conditioned squeezing gate :math:`C_xS(\beta)`

    This gate implements the unitary

    .. math::

        CS(\beta) &= \exp\left[\frac{1}{2}\sigma_x (\beta^* a^2 - \beta (\ad)^2)\right]

    which differs from :class:`~hybridlane.ops.ConditionalSqueezing` due to the :math:`\sigma_x` factor
    instead of :math:`\sigma_z`.

    This is represented by the hardware instruction ``RampUp``, and it can only be used on hardware qumode ``a0m1``
    """

    num_params = 1
    num_wires = 2
    num_qumodes = 1
    grad_method = "F"

    resource_keys = set()

    def __init__(
        self,
        ratio: TensorLike,
        wires: WiresLike,
        id: str | None = None,
    ):
        r"""
        Args:
            ratio: The blue/red ratio
        """
        super().__init__(ratio, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "RampUp", cache=cache
        )


class SidebandProbe(Operation, Hybrid):
    r"""General sideband probe operation

    This is represented by the hardware instruction ``Rt_SBProbe``
    """

    num_params = 4
    num_wires = 2
    num_qumodes = 1
    grad_method = None

    resource_keys = set()

    def __init__(
        self,
        duration_us: TensorLike,
        phase: TensorLike,
        sign: TensorLike,
        detuning: TensorLike,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(duration_us, phase, sign, detuning, wires=wires, id=id)

    def pow(self, n: int | float):
        duration, *params = self.data
        return [SidebandProbe(duration * n, *params, wires=self.wires)]

    @property
    def resource_params(self):
        return {}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "Rt_SBP", cache=cache
        )


class FockStatePrep(Operation, Hybrid):
    r"""Prepare a definite Fock state

    This is identical to ``hqml.FockLadder`` except it's only supported on hardware qumodes ``a0m1`` and ``a1m1``.
    This gate is represented in terms of a native Jaqal instruction ``FockState``.
    """

    num_params = 1
    num_wires = 2
    num_qumodes = 1
    grad_method = None

    resource_keys = set()

    def __init__(self, n: int, wires: WiresLike = None, id: str | None = None):
        super().__init__(n, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}


class NativeBeamsplitter(Operation, Hybrid):
    r"""Hardware-native beamsplitter gate

    This class is named NativeBeamsplitter to distinguish it from :class:`~hybridlane.ops.Beamsplitter`,
    as it has different arguments. It is represented by the hardware instruction ``Beamsplitter``.

    Currently this gate can only be executed on the tilt modes (``a0m1``, ``a1m1``)
    """

    num_params = 4
    num_wires = 3
    num_qumodes = 2
    grad_method = None

    resource_keys = set()

    def __init__(
        self,
        detuning1: TensorLike,
        detuning2: TensorLike,
        duration: TensorLike,
        phase: TensorLike,
        wires: WiresLike = None,
        id: str | None = None,
    ):
        super().__init__(detuning1, detuning2, duration, phase, wires=wires, id=id)

    @property
    def resource_params(self):
        return {}

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "BS", cache=cache
        )


class R(Operation):
    r"""Rotation about an axis :math:`R_{\phi}(\theta)`

    .. math::

        R_{\phi}(\theta) = e^{-i\theta/2 (\cos\phi X + \sin\phi Y)}
    """

    num_params = 2
    ndim_params = (0, 0)
    num_wires = 1

    resource_keys = set()

    def __init__(self, theta, phi, wires: WiresLike = None, id: str | None = None):
        super().__init__(theta, phi, wires=wires, id=id)

    def adjoint(self):
        return R(-self.data[0], self.data[1], wires=self.wires)

    def pow(self, z):
        return [R(z * self.data[0], self.data[1], wires=self.wires)]

    @property
    def resource_params(self):
        return {}

    def simplify(self):
        theta, phi = self.data[0] % (4 * math.pi), self.data[1] % math.pi

        if _can_replace(theta, 0):
            return qml.Identity(wires=self.wires)

        elif _can_replace(phi, 0):
            return qml.RX(theta, wires=self.wires)

        elif _can_replace(phi, math.pi / 2):
            return qml.RY(theta, wires=self.wires)

        elif _can_replace(phi, -math.pi):
            return qml.RX(-theta, wires=self.wires)

        elif _can_replace(phi, -math.pi / 2):
            return qml.RY(theta, wires=self.wires)

        return R(theta, phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "R", cache=cache
        )
