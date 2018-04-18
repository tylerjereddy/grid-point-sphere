import pytest
import numpy as np
from math import sqrt
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

def test_grid_cells_level_1_shape():
    # test that the number of grid cells
    # iterated over in level 1 matches
    # with expectations

    m_lambda_1 = 10
    m_phi_1 = 20

    # based on back of the envelope
    # calculations, I expect
    cells_expected = (m_lambda_1 - 1) * (m_phi_1 - 1)

    test_polygon = np.array([[0,0,1],
                             [0,1,0],
                             [1,0,0]])

    result = lib.cast_subgrids(test_polygon).size

    assert result == cells_expected

def test_grid_level_1_edge_count_boundary():
    # given a test input spherical polygon
    # that covers less than 1/8 the total
    # surface area of a sphere, it should
    # be impossible to have spherical polygon
    # edges in >= 1/2 of all grid cells
    # covering the sphere surface at level 1

    test_polygon = np.array([[0,0,1],
                             [0,np.sqrt(2) / 2., np.sqrt(2) / 2.],
                             [np.sqrt(2) / 2., 0, np.sqrt(2) / 2.]])

    result = lib.cast_subgrids(test_polygon)
    max_allowed = int(result.size / 2.)

    assert np.count_nonzero(result) < max_allowed
