# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Device implementation for Bosonic Qiskit.

Using this module requires ``bosonic-qiskit`` to be installed. You can do this with pip or it's
included automatically when installing ``hybridlane[bq]``.
"""

from .device import BosonicQiskitDevice

__all__ = ["BosonicQiskitDevice"]
