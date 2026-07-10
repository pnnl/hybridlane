# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Utility module for working with Bosonic Qiskit's registers."""

import math
from collections import OrderedDict
from collections.abc import Hashable, Mapping

import bosonic_qiskit as bq
import qiskit as qk
from pennylane.wires import Wires
from qiskit.circuit import Qubit

from ... import wires as sa
from ...measurements import FockTruncation

QumodeType = list[Qubit]


class RegisterMapping(Mapping):
    r"""Utility class to map wires to bosonic qiskit registers."""

    axes_map: dict[Hashable, tuple[int, ...]]

    def __init__(
        self,
        sa_result: sa.TypeCheckResult,
        fock_truncation: FockTruncation,
    ):
        r"""Constructs the mapping.

        Args:
            sa_result: The result of calling ``hl.type_check`` on the circuit

            fock_truncation: The truncation to use for the qumode dimensions.
        """
        self._truncation = fock_truncation
        self.sa_res = sa_result
        self.mapping = self._prepare(sa_result)

    def _prepare(self, sa_result: sa.TypeCheckResult) -> OrderedDict[Hashable, Qubit | QumodeType]:
        mapping = OrderedDict()
        self.axes_map = {}

        # Put all qubits into the same register
        self.qubit_reg = qk.QuantumRegister(len(sa_result.qubits), name="q")
        for i, wire in enumerate(sa_result.qubits):
            mapping[wire] = self.qubit_reg[i]
            self.axes_map[wire] = (i,)

        # Here we just make a unique register for each qumode. One could also consider grouping
        # them by truncation and then putting all qumodes with the same truncation into the same
        # register
        self.qumode_regs = []
        total_num_qubits_created = len(self.qubit_reg)
        for i, wire in enumerate(sa_result.qumodes):
            try:
                dim = self._truncation.dim(wire)
                required_qubits = math.ceil(math.log2(dim))
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
                raise RuntimeError(f"Need to specify a truncation for qumode `{wire}`") from e

        return mapping

    @property
    def wire_order(self):
        r"""The order of the wires in the hybridlane tape."""
        return Wires.all_wires(self.mapping.keys())

    @property
    def regs(self):
        r"""The list of registers of the BQ circuit."""
        return [self.qubit_reg, *self.qumode_regs]

    @property
    def truncation(self):
        r"""The truncation used for the qumodes."""
        return self._truncation

    def __len__(self):
        return len(self.mapping)

    def __iter__(self):
        return iter(self.mapping)

    def __getitem__(
        self, wire_or_wires: Hashable | Wires
    ) -> Qubit | QumodeType | list[Qubit | QumodeType]:
        if isinstance(wire_or_wires, Wires):
            # Unbatch wires if possible
            if len(wire_or_wires) == 1:
                return self.mapping[wire_or_wires[0]]

            return [self.mapping[w] for w in wire_or_wires]

        return self.mapping[wire_or_wires]
