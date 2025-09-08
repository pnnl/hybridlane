# Copyright (c) 2025, Battelle Memorial Institute

# This software is licensed under the 2-Clause BSD License.
# See the LICENSE.txt file for full license text.
from .base import SampleMeasurement


class CountsMP(SampleMeasurement):
    # todo: discuss about how to implement counts
    # for any system measuring in phase space, we necessarily can't bin the outcomes without
    # discretizing the x dimension. the counts dictionary would be no more compact than it started
    pass
