# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause

r"""Module containing the native bosonic gates of the ion trap"""

import math

import pennylane as qp
from pennylane.operation import Operation
from pennylane.typing import TensorLike
from pennylane.wires import WiresLike

import hybridlane as hl

from ...math.utils import can_replace, concrete_or_error
from ...ops.mixins import HybridOperation

Red = hl.Red
Blue = hl.Blue
XCD = hl.XCD
YCD = hl.YCD
ZCD = hl.CD
FockState = hl.FockState


class ConditionalXSqueezing(HybridOperation):
    r"""Qubit-conditioned squeezing gate :math:`xCS(\beta)`

    This gate implements the unitary

    .. math::

        xCS(\beta) = \exp\left[\frac{1}{2}\sigma_x (\beta^* a^2 - \beta (\ad)^2)\right]

    which differs from :class:`~hybridlane.ops.ConditionalSqueezing` due to the
    :math:`\sigma_x` factor instead of :math:`\sigma_z`.

    This is represented by the hardware instruction ``RampUp``, and it can only be used
    on hardware qumode ``m1i1``
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


class SidebandProbe(HybridOperation):
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


class NativeBeamsplitter(HybridOperation):
    r"""Hardware-native beamsplitter gate

    This class is named NativeBeamsplitter to distinguish it from
    :class:`~hybridlane.ops.Beamsplitter`, as it has different arguments. It is
    represented by the hardware instruction ``Beamsplitter``.

    Currently this gate can only be executed on the tilt modes (``m0i1``, ``m1i1``)
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

        theta = concrete_or_error(
            None, theta, "Cannot simplify R when ``theta`` is a tracer"
        )
        phi = concrete_or_error(None, phi, "Cannot simplify R when ``phi`` is a tracer")
        if can_replace(theta, 0):
            return qp.Identity(wires=self.wires)

        elif can_replace(phi, 0):
            return qp.RX(theta, wires=self.wires)

        elif can_replace(phi, math.pi / 2):
            return qp.RY(theta, wires=self.wires)

        elif can_replace(phi, -math.pi):
            return qp.RX(-theta, wires=self.wires)

        elif can_replace(phi, -math.pi / 2):
            return qp.RY(theta, wires=self.wires)

        return R(theta, phi, wires=self.wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(
            decimals=decimals, base_label=base_label or "R", cache=cache
        )
