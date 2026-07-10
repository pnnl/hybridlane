# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Patches for PennyLane's graph decomposition system."""

from unittest.mock import patch

import pennylane as qp
from pennylane.decomposition import CompressedResourceOp, DecompositionRule
from pennylane.decomposition import DecompositionGraph as PennyLaneDecompositionGraph
from typing_extensions import override

import hybridlane as hl

from ..ops.op_math.decompositions.qubit_conditioned_decompositions import (
    decompose_multi_qcond,
)
from .symbolic_decomposition import ctrl_from_qcond, flip_pow_qcond, make_qcond_decomp


class DecompositionGraph(PennyLaneDecompositionGraph):
    r"""Custom decomposition graph for PennyLane supporting our custom symbolic operators."""

    @override
    def _get_decompositions(
        self, op: CompressedResourceOp, use_reconstructor: bool = False
    ) -> list[DecompositionRule]:
        decomps = super()._get_decompositions(op, use_reconstructor)

        if op.op_type in (hl.ops.QubitConditioned,):
            decomps.extend(self._get_qubit_conditioned_decompositions(op, use_reconstructor))

        return decomps

    @override
    def _get_controlled_decompositions(
        self, op: CompressedResourceOp, use_reconstructor: bool = False
    ) -> list[DecompositionRule]:
        decomps = super()._get_controlled_decompositions(op, use_reconstructor)

        # Can generally synthesize the controlled version from conditional gates
        decomps.append(ctrl_from_qcond)

        return decomps

    def _get_qubit_conditioned_decompositions(
        self, op: CompressedResourceOp, use_reconstructor: bool = False
    ) -> list[DecompositionRule]:
        base_class, base_params = (
            op.params["base_class"],
            op.params["base_params"],
        )

        # General case is to apply qcond to each gate in the decomposition
        base = qp.resource_rep(base_class, **base_params)
        rules = [
            make_qcond_decomp(decomp)
            for decomp in self._get_decompositions(base, use_reconstructor)
        ]

        # Can always reduce to 1 condition qubit
        rules.append(decompose_multi_qcond)

        return rules

    @override
    @staticmethod
    def _get_pow_decompositions(
        op: CompressedResourceOp, use_reconstructor: bool = False
    ) -> list[DecompositionRule]:
        decomps = PennyLaneDecompositionGraph._get_pow_decompositions(op)

        if op.params["base_class"] in (hl.ops.QubitConditioned,):
            decomps.append(flip_pow_qcond)

        return decomps


_ = patch("pennylane.transforms.decompose.DecompositionGraph", DecompositionGraph).start()
