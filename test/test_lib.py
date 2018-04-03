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
                          np.array([-sqrt(2) / 2., sqrt(2) / 2., 0]),
                          np.array([sqrt(2) / 2., sqrt(2) / 2., 0]),
                          np.zeros((3,)),
                          True),
                         # adjusting the first arc such that it
                         # no longer traverses the equator means
                         # there should be no arc intersection
                         (np.array([0, sqrt(2) / 2., sqrt(2) / 2.]),
                          np.array([0, sqrt(3) / 2., 0.5]),
                          np.array([-sqrt(2) / 2., sqrt(2) / 2., 0]),
                          np.array([sqrt(2) / 2., sqrt(2) / 2., 0]),
                          np.zeros((3,)),
                          False),
                          ])
def test_arc_intersect(A, B, C, D, center, expected):
    # test that spherical arc (parts
    # of great circles) intersections
    # are properly determined
    result = lib.determine_arc_intersection(A, B, C, D, center)
    assert result == expected

def test_grid_cast_level_1_shape():
    # test that the meshgrid data structure
    # cast onto the sphere for the first level
    # has the appropriate (always fixed)
    # shape / resolution

    # the resolution is fixed in the
    # manuscript as follows:
    m_lambda_1 = 10
    m_phi_1 = 20

    level_1_grid = lib.cast_grid_level_1()
    result = level_1_grid.shape

    assert result == (2, m_lambda_1, m_phi_1)
