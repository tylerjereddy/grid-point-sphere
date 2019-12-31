import pytest
import numpy as np
from numpy.testing import (assert_almost_equal,
                           assert_allclose)
from math import sqrt
import lib

@pytest.mark.parametrize("A, B, C, center, expected", [
    # define a plane that divides unit sphere into
    # East / West "hemispheres"
    # then probe points at extreme East / West

    # NOTE: I'm not sure if human intuition for
    # "right" vs. "left" here make sense--after all,
    # you could just rotate the sphere perspective, but
    # we at least test for consistency in the code
    (np.array([0, 1, 0]),
     np.array([0, 0, 1]),
     np.array([1, 0, 0]), # query point on "left"
     np.zeros((1, 3)),
     "left"),
    # checking the antipode of the query point should
    # produce a "right" side result (opposite of above)
    (np.array([0, 1, 0]),
     np.array([0, 0, 1]),
     np.array([-1, 0, 0]), # query point on "right"
     np.zeros((1, 3)),
     "right"),
    # with a basic "orientation" for left / right relative
    # to viewer established above, move the query point to
    # "front right"
    (np.array([0, 1, 0]),
     np.array([0, 0, 1]),
     np.array([-np.sqrt(2) / 2., 
               np.sqrt(2) / 2., 0]),
     np.zeros((1, 3)),
     "right"),
    # same for "back left"
    (np.array([0, 1, 0]),
     np.array([0, 0, 1]),
     np.array([np.sqrt(2) / 2., 
               -np.sqrt(2) / 2., 0]),
     np.zeros((1, 3)),
     "left"),
    ])
def test_arc_plane_side(A, B, C, center, expected):
    # need to be able to reliably determine
    # which side of a great circle plane a point
    # on the surface of a sphere is on
    actual = lib.arc_plane_side(center, A, B, C)
    if expected == 'right':
        assert actual < 0
    else:
        assert actual > 0

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

    result = lib.cast_subgrids(test_polygon)[0].size

    assert result == cells_expected

def test_grid_level_1_edge_count_boundary():
    # given a test input spherical polygon
    # that covers less than 1/8 the total
    # surface area of a sphere, it should
    # be impossible to have spherical polygon
    # edges in >= 1/8 of all grid cells
    # covering the sphere surface at level 1

    test_polygon = np.array([[0,0,1],
                             [0,np.sqrt(2) / 2., np.sqrt(2) / 2.],
                             [np.sqrt(2) / 2., 0, np.sqrt(2) / 2.]])

    result = lib.cast_subgrids(test_polygon)[0]
    max_allowed = int(result.size / 2.)

    assert np.count_nonzero(result) < max_allowed

    # because the input spherical polygon is a spherical triangle,
    # there are three vertices, and therefore a maximum of three level 1
    # grid cells permitted to contain > 1 spherical polygon edge
    assert (result == 2).sum() <= 3

    # no level 1 grid cell should contain > 2 spherical polygon edges
    assert (result > 2).sum() == 0

@pytest.mark.parametrize("i, j, k, l_lambda, l_phi, N", [
                         (2, 3, 1.0, 2.5, 2.5, 10),
                         (4, 9, 1.0, 4.5, 9.5, 200),
                         (4, 9, 1.0, 4.5, 9.5, 0),
                         ])
class TestGridSubdivisions(object):

    def test_calc_m_lambda(self, i, j, k, l_lambda, l_phi, N):
        # since the manuscript defines m_lambda
        # as the number of cells in the latitude
        # direction, we should at least ensure
        # that the result with reasonable
        # input is always an integer number
        # of cells
        result = lib.calc_m_lambda(i=i,
                                   j=j,
                                   l_lambda=l_lambda,
                                   l_phi=l_phi,
                                   N=N)
        assert isinstance(result, int)

    # similarly for m_phi
    def test_calc_m_phi(self, i, j, k, l_lambda, l_phi, N):
        result = lib.calc_m_phi(i=i,
                                j=j,
                                l_lambda=l_lambda,
                                l_phi=l_phi,
                                N=N)
        assert isinstance(result, int)

@pytest.mark.parametrize("first_cell_lat_1,"
                        "first_cell_lat_2,"
                        "first_cell_long_1,"
                        "first_cell_long_2,"
                        "list_edges_in_first_cell,"
                        "center,"
                        "radius,"
                        "expected", [
                         # lat = 0, lon = 90 is +y axis of
                         # sphere facing viewer with x = -1
                         # on right and x = +1 on left
                         # or 180 lon on right and 0 lon on left
                         # +90 lat is +1 Z (up)

                         # careful with clockwise and
                         # CCW internal conventions!!!
                         # walking CCW along polygon edges, inside means on your
                         # left!

                         # the first case divides
                         # the sphere into 16
                         # grid cells, picks one
                         # of these cells & encloses
                         # the center point of that cell
                         # in a small spherical triangle
                         # as defined by all 3 of its arcs
                         # so, the function should determine
                         # that the center point is inside
                         # the spherical polygon

                         (0, 45, 90, 180,
                         [np.array([[10, 90],
                                    [10, 180]]),
                          np.array([[10, 180],
                                    [65, 135]]),
                          np.array([[65, 135],
                                    [10, 90]])],
                          np.zeros(3,),
                          1.0,
                          'inside'),
                         # for the second case, use
                         # the same system, but move
                         # the tip of the spherical
                         # triangle such that the
                         # grid center point is
                         # excluded
                         (0, 45, 90, 180,
                         [np.array([[10, 90],
                                    [10, 180]]),
                          # use a small rise
                          # to exclude grid center
                          np.array([[10, 180],
                                    [15, 135]]), # under center
                          np.array([[15, 135],
                                    [10, 90]])],
                          np.zeros(3,),
                          1.0,
                          'outside'),
                         # need a test case where O_i is
                         # on the right side of AB and
                         # outside the spherical polygon
                         (0, 45, 90, 180,
                         [np.array([[10, 90],
                                    [65, 110]]),
                          np.array([[65, 110],
                                    [20, 45]]),
                          np.array([[20, 45],
                                    [10, 90]])],
                          np.zeros(3,),
                          1.0,
                          'outside'),
                         # and now with O_i to the right
                         # of AB and inside the spherical polygon
                         (0, 45, 90, 180,
                         [np.array([[10, 90],
                                    [50, 120]]),
                         # intersect Oi_C line, trigger inclusion
                          np.array([[50, 120],
                                    [10, 120 ]])],
                          np.zeros(3,),
                          1.0,
                          'inside'),
                         ])
def test_first_traversal_determination(first_cell_lat_1,
                                       first_cell_lat_2,
                                       first_cell_long_1,
                                       first_cell_long_2,
                                       list_edges_in_first_cell,
                                       center,
                                       radius,
                                       expected):
    # NOTE: it seems that we need to use the
    # "bottom left" grid cell (that contains
    # a spherical polygon edge) for this function
    # to work properly, although this does not
    # seem to be articulated clearly in the article
    # this limits the range of valid test cases
    # that may be used above

    # test for the function that implements
    # the algorithm depicted in Figure 4
    # of the manuscript

    # verify proper determination of a point
    # inside / outside a spherical polygon
    # based on various arcs in the grid cell
    # this approach is used only for the first
    # grid cells parsed on a given grid
    actual = lib.determine_first_traversal_point(first_cell_lat_1,
                                                 first_cell_lat_2,
                                                 first_cell_long_1,
                                                 first_cell_long_2,
                                                 list_edges_in_first_cell,
                                                 center,
                                                 radius)
    assert actual == expected

@pytest.mark.parametrize("center_1_property,"
                         "intersection_count,"
                         "expected", [
                         ('inside', 1, 'outside'),
                         ('inside', 8, 'inside'),
                         ('outside', 3, 'inside'),
                         ('outside', 6, 'outside'),
                         ])
def test_inclusion_property(center_1_property,
                            intersection_count,
                            expected):
    result = lib.inclusion_property(center_1_property=center_1_property,
                                    intersection_count=intersection_count)
    assert result == expected

class TestGridCenterPoint(object):
    # tests for grid_center_point() function

    @pytest.mark.parametrize("long_1, long_2, lat_1, lat_2", [
                              (-181, -179, 45, 51),
                              (55, 70, 89, 92),
                              ])
    def test_limits(self, long_1, long_2,
                    lat_1, lat_2):
        # the manuscript clearly indicates that
        # latitude is in [-90, 90]
        # longitude is in [-180, 180]
        # so grid_center_point should raise an
        # appropriate exception if attempting to
        # operate outside these bounds
        with pytest.raises(ValueError):
            lib.grid_center_point(grid_cell_long_1=long_1,
                                  grid_cell_long_2=long_2,
                                  grid_cell_lat_1=lat_1,
                                  grid_cell_lat_2=lat_2)

    @pytest.mark.parametrize("long_1, long_2, lat_1, lat_2", [
                              (180, 0, 0, 0),
                              (-180, 0, 30, 30),
                              (-90, 90, 20, 20),
                              (0, 0, 90, -90),
                              ])
    def test_antipode_handling(self, long_1, long_2, lat_1, lat_2):
        # an appropriate error should be raised if trying to handle
        # antipodes, for which the midpoint would be ambiguous
        with pytest.raises(ValueError):
            lib.grid_center_point(grid_cell_long_1=long_1,
                                  grid_cell_long_2=long_2,
                                  grid_cell_lat_1=lat_1,
                                  grid_cell_lat_2=lat_2)


    @pytest.mark.parametrize("long_1, long_2, lat_1, lat_2, expected", [
                              (-180, -174, 40, 50,
                              np.array([45, -177])),
                              (178, -178, 40, 50,
                              np.array([45, 180])),
                              (5, -5, 40, 50,
                              np.array([45, 0])),
                              (-20, 90, 0, 0,
                              np.array([0, 35])),
                              ])
    def test_centers(self, long_1, long_2,
                           lat_1, lat_2,
                           expected):
        # verify that grid_center_point() can correctly
        # determine center values of grid cells for a variety
        # of cases
        actual = lib.grid_center_point(grid_cell_long_1=long_1,
                                       grid_cell_long_2=long_2,
                                       grid_cell_lat_1=lat_1,
                                       grid_cell_lat_2=lat_2)

        assert_almost_equal(actual, expected)

def test_level_1_grid_centers():
    # check some simple properties of the level 1
    # grid center data structure produced by
    # produce_level_1_grid_centers()

    # since the L1 grid is fixed, the choice
    # of input spherical_polygon isn't too
    # important here
    spherical_polygon = np.array([[0, 1, 0],
                                  [0, 0, 1],
                                  [-1, 0, 0]], dtype=np.float64)

    # retrieve the edge counts for level 1
    # from another function, so that we
    # have a reference data structure for
    # shape comparison
    expected_edge_count_array_L1 = lib.cast_subgrids(spherical_polygon)[0]

    (actual_grid_cell_center_coords_L1,
     actual_edge_count_array_L1) = lib.produce_level_1_grid_centers(
                                            spherical_polygon)

    # the edge count array should have been
    # passed through produce_level_1_grid_centers()
    # unchanged
    assert_allclose(actual_edge_count_array_L1,
                    expected_edge_count_array_L1)

    # the number of grid cell center coordinates
    # should match the edge count data structure size
    assert_allclose(actual_grid_cell_center_coords_L1.shape[0],
                    expected_edge_count_array_L1.size)

    # all L1 grid cell center coords should be on the unit
    # sphere/norm
    norms = np.linalg.norm(actual_grid_cell_center_coords_L1, axis=1)
    expected_norms = np.ones((expected_edge_count_array_L1.size,))
    assert_allclose(norms, expected_norms)
