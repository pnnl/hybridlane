# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
import re
from unittest.mock import patch

from pennylane.decomposition.utils import translate_op_alias as _old_translate_op_alias

# Fix pennylane's translate_op_alias function to support our custom symbolic operators


def translate_op_alias(op_alias):
    if match := re.match(r"qCond\((\w+)\)", op_alias):
        base_op_name = match.group(1)
        return f"qCond({translate_op_alias(base_op_name)})"

    return _old_translate_op_alias(op_alias)


patch(
    "pennylane.decomposition.decomposition_rule.translate_op_alias", translate_op_alias
).start()
patch(
    "pennylane.decomposition.decomposition_graph.translate_op_alias", translate_op_alias
).start()
patch("pennylane.transforms.decompose.translate_op_alias", translate_op_alias).start()
