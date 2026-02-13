# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from typing_extensions import override
from unittest.mock import patch

import pennylane as qml
from pennylane.decomposition import CompressedResourceOp, DecompositionRule
from pennylane.decomposition import DecompositionGraph as PLDG

import hybridlane as hqml

from ..ops.op_math.decompositions.qubit_conditioned_decompositions import (
    decompose_multi_qcond,
)
from .symbolic_decomposition import ctrl_from_qcond, flip_pow_qcond, make_qcond_decomp


class DecompositionGraph(PLDG):
    @override
    def _get_decompositions(
        self, op_node: CompressedResourceOp
    ) -> list[DecompositionRule]:
        decomps = super()._get_decompositions(op_node)

        if op_node.op_type in (hqml.ops.QubitConditioned,):
            decomps.extend(self._get_qubit_conditioned_decompositions(op_node))

        return decomps

    @override
    def _get_controlled_decompositions(
        self, op_node: CompressedResourceOp
    ) -> list[DecompositionRule]:
        decomps = super()._get_controlled_decompositions(op_node)

        # Can generally synthesize the controlled version from conditional gates
        decomps.append(ctrl_from_qcond)

        return decomps

    def _get_qubit_conditioned_decompositions(
        self, op_node: CompressedResourceOp
    ) -> list[DecompositionRule]:
        base_class, base_params = (
            op_node.params["base_class"],
            op_node.params["base_params"],
        )

        # General case is to apply qcond to each gate in the decomposition
        base = qml.resource_rep(base_class, **base_params)
        rules = [make_qcond_decomp(decomp) for decomp in self._get_decompositions(base)]

        # Can always reduce to 1 condition qubit
        rules.append(decompose_multi_qcond)

        return rules

    @override
    def _get_pow_decompositions(
        self, op_node: CompressedResourceOp
    ) -> list[DecompositionRule]:
        decomps = super()._get_pow_decompositions(op_node)

        if op_node.params["base_class"] in (hqml.ops.QubitConditioned,):
            decomps.append(flip_pow_qcond)

        return decomps


_ = patch(
    "pennylane.transforms.decompose.DecompositionGraph", DecompositionGraph
).start()
