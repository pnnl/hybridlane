# SPDX-FileCopyrightText: 2025 Battelle Memorial Institute
# SPDX-License-Identifier: BSD-2-Clause
r"""Patches for PennyLane's decomposition utils module."""

import re
from functools import singledispatch
from unittest.mock import patch

from pennylane.decomposition.utils import to_name as pl_to_name
from pennylane.decomposition.utils import translate_op_alias as _old_translate_op_alias
from pennylane.operation import Operator

# Fix pennylane's translate_op_alias function to support our custom symbolic operators


@singledispatch
def to_name(op):
    r"""Returns the canonical name of an operator in the graph decomposition framework."""
    return pl_to_name(op)


@to_name.register
def _op_to_name(op: Operator):
    return translate_op_alias(op.name)


@to_name.register
def _str_to_name(op: str):
    return translate_op_alias(op)


@to_name.register
def _type_to_name(op: type):
    return translate_op_alias(op.__name__)


def translate_op_alias(op_alias):
    r"""Translates an operator alias to its canonical name."""
    if match := re.match(r"(?:qCond|QubitConditioned)\((\w+)\)", op_alias):
        base_op_name = match.group(1)
        return f"qCond({translate_op_alias(base_op_name)})"

    return _old_translate_op_alias(op_alias)


patch("pennylane.decomposition.decomposition_rule.to_name", to_name).start()
patch("pennylane.decomposition.resources.to_name", to_name).start()
patch("pennylane.decomposition.decomposition_graph.to_name", to_name).start()
patch("pennylane.decomposition.gate_set.to_name", to_name).start()
