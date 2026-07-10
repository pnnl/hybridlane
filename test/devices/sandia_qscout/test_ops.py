# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
import hybridlane as hl
from hybridlane.devices.sandia_qscout import ops
from hybridlane.wires.type_check import infer_wires


class TestR:
    def test_type_check(self):
        op = ops.R(0.123, 0.456, wires=0)
        context = infer_wires(op, {})
        assert context == {0: hl.Qubit()}
