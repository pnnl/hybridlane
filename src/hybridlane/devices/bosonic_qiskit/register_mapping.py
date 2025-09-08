# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import math
from collections import OrderedDict
from typing import Hashable, Mapping, Union

import c2qa as bq
import qiskit as qk
from pennylane.wires import Wires
from qiskit.circuit import Qubit

from ... import sa
from ...measurements import FockTruncation

QumodeType = list[Qubit]


# Todo: Currently this only handles fock truncations because that's what bosonic qiskit does. Should
# we refactor to use general truncations?
class RegisterMapping(Mapping):
    r"""Utility class to map wires -> bosonic qiskit registers"""

    axes_map: dict[Hashable, tuple[int, ...]]

    def __init__(
        self,
        sa_result: sa.StaticAnalysisResult,
        fock_truncation: FockTruncation,
    ):
        self._truncation = fock_truncation
        self.mapping = self._prepare(sa_result)

    def _prepare(
        self, sa_result: sa.StaticAnalysisResult
    ) -> OrderedDict[Hashable, Union[Qubit, QumodeType]]:
        mapping = OrderedDict()
        self.axes_map = {}

        # Put all qubits into the same register
        self.qubit_reg = qk.QuantumRegister(len(sa_result.qubits), name="q")
        for i, wire in enumerate(sa_result.qubits):
            mapping[wire] = self.qubit_reg[i]
            self.axes_map[wire] = (i,)

        # Here we just make a unique register for each qumode. One could also consider grouping them by
        # truncation and then putting all qumodes with the same truncation into the same register
        self.qumode_regs = []
        total_num_qubits_created = len(self.qubit_reg)
        for i, wire in enumerate(sa_result.qumodes):
            try:
                dim = self._truncation.dim(wire)
                required_qubits = int(math.ceil(math.log2(dim)))
                qmreg = bq.QumodeRegister(1, required_qubits, name=f"m{i}")
                mapping[wire] = qmreg[0]
                self.qumode_regs.append(qmreg)
                self.axes_map[wire] = tuple(
                    range(
                        total_num_qubits_created,
                        total_num_qubits_created + required_qubits,
                    )
                )
                total_num_qubits_created += required_qubits
            except KeyError as e:
                raise RuntimeError(
                    f"Need to specify a truncation for qumode `{wire}`"
                ) from e

        return mapping

    @property
    def wire_order(self):
        return Wires.all_wires(self.mapping.keys())

    @property
    def regs(self):
        return [self.qubit_reg] + self.qumode_regs

    @property
    def truncation(self):
        return self._truncation

    def __len__(self):
        return len(self.mapping)

    def __iter__(self):
        return iter(self.mapping)

    def __getitem__(
        self, wire_or_wires: Hashable | Wires
    ) -> Union[Qubit, QumodeType] | list[Union[Qubit, QumodeType]]:
        if isinstance(wire_or_wires, Wires):
            # Unbatch wires if possible
            if len(wire_or_wires) == 1:
                return self.mapping[wire_or_wires[0]]

            return [self.mapping[w] for w in wire_or_wires]

        return self.mapping[wire_or_wires]
