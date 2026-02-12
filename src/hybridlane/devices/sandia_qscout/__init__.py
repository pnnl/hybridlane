# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.

r"""Module for all QSCOUT-related functionality :footcite:p:`clark2021engineering`

Device details (``sandiaqscout.hybrid``)
----------------------------------------

The device supports up to 6 qubits (ions) and their associated motional modes. By
default, the center-of-mass (COM) modes are disabled due to their higher noise
levels, but they can be enabled by setting the ``enable_com_modes`` option to
``True`` when initializing the device. Thus for a circuit with :math:`n` qubits,
there are :math:`2n-2` motional modes available by default, or :math:`2n` if the COM
modes are enabled.

**Wires** The device supports hardware wires and virtual wires. Hardware qubits are
addressed with integers :math:`0` to :math:`5`, while motional modes are addressed
using the :class:`~hybridlane.jaqal.Qumode` object or strings of the form
``"m{manifold}i{index}"``, where ``manifold`` is ``1`` for the lower motional
manifold and ``0`` for the upper manifold, and ``index`` is the index of the mode.

Example with hardware wires:

.. code:: python

    dev = qml.device("sandiaqscout.hybrid", use_virtual_wires=False, n_qubits=4)

    @qml.set_shots(10)
    @qml.qnode(dev)
    def circuit():
        hqml.FockState(3, [0, "m1i2"])
        return hqml.expval(qml.Z(0))

    print(qml.draw(circuit)())

When using hardware wires, the user is responsible for ensuring that gates adhere to
any constraints. Additionally, for optimal performance, the qubits and qumodes should
be chosen to maximize coupling strengths to reduce gate time. The lower manifold (1)
has stronger couplings.

By default, the device uses virtual wire allocation to assign physical wires to
virtual wires based on constraints of the gates in the circuit. This can be
disabled by setting ``use_virtual_wires`` to ``False`` when initializing the
device, in which case the circuit must use only physical wires.

Example with virtual wires:

.. code:: python

    qml.decomposition.enable_graph()
    dev = qml.device("sandiaqscout.hybrid", n_qubits=4)

    @qml.set_shots(10)
    @qml.qnode(dev)
    def circuit():
        hqml.FockState(3, ["q", "m"])
        return hqml.expval(qml.Z("q"))

    print(qml.draw(circuit, level="device")())

Note that virtual wire allocation does not yet perform any ranking for valid solutions
or noise-aware compilation.

**Native gates**: The native gate set includes common qubit gates and some hybrid gates,
particularly implementing the Sideband ISA :footcite:p:`liu2026hybrid`. The native
gates are available in :mod:`hybridlane.devices.sandia_qscout.ops` and currently
include:

- **Qubit gates**: :math:`R_\phi, R_x, R_y, R_z, S, S^\dagger, S_x, S_x^\dagger, XX, YY, ZZ`
- **Hybrid gates**: :math:`JC, AJC, xCD, xCS, BS`

References
----------

.. footbibliography::
"""

from . import ops
from .device import QscoutIonTrap, get_compiler
from .draw import get_default_style
from .jaqal import Qumode, batch_to_jaqal, to_jaqal

__all__ = [
    "ops",
    "QscoutIonTrap",
    "get_compiler",
    "get_default_style",
    "batch_to_jaqal",
    "to_jaqal",
    "Qumode",
]
