import pytest
import numpy as np
from math import sqrt
import sys
sys.path.append('..')
import lib

@pytest.mark.parametrize("A, B, C, D, center, expected", [
                         # a North to South arc through + y
                         # should intersect a West to East arc
                         # through +y
                         (np.array([0, sqrt(2) / 2., sqrt(2) / 2.]),
                          np.array([0, sqrt(2) / 2., -sqrt(2) / 2.]),
                          np.array([-sqrt(2) / 2., -sqrt(2) / 2., 0]),
                          np.array([sqrt(2) / 2., sqrt(2) / 2., 0]),
                          np.zeros((3,)),
                          True),
                          ])
def test_arc_intersect(A, B, C, D, center, expected):
    # test that spherical arc (parts
    # of great circles) intersections
    # are properly determined
    result = lib.determine_arc_intersection(A, B, C, D, center)
    assert result == expected
