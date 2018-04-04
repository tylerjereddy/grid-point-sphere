'''Library of utility functions for the point in spherical polygon
algorithm implementation described by Li et al. (2017).
'''

import numpy as np

def calc_m_lambda(i, j, k=1.0, l_lambda=1, l_phi=1, N=0, MAXD=4):
    '''
    From equation 1 in Li et al. (2017)

    i is the level of the grid

    j is the grid cell on the level designated by i

    k is a coefficient empirically set to 1
    by authors

    l_lambda is the degree of latitude spanned by
    the grid cell j, at level i

    l_phi is the degree of longitude spanned by
    the grid cell j, at level i

    MAXD is the set maximum level of the
    multilevel grids

    N is the number of spherical polygon
    edges in the grid cell j at level i

    Returns m_lambda -- the number of cells for the
    grid in the latitude (N-S) direction at level
    i, cell j.
    '''
    if i == 1 and j == 1:
        # use a fixed low resolution for
        # the first level grid
        m_lambda = 10
    else:
        m_lambda = l_lambda * _grid_build_coef(k,
                                               l_lambda,
                                               l_phi,
                                               N)

    return m_lambda

def calc_m_phi(i, j, k=1.0, l_lambda=1, l_phi=1, N=0, MAXD=4):
    '''
    From equation 1 in Li et al. (2017)

    i is the level of the grid

    j is the grid cell on the level designated by i

    k is a coefficient empirically set to 1
    by authors

    l_lambda is the degree of latitude spanned by
    the grid cell j, at level i

    l_phi is the degree of longitude spanned by
    the grid cell j, at level i

    MAXD is the set maximum level of the
    multilevel grids

    N is the number of spherical polygon
    edges in the grid cell j at level i

    Returns m_phi -- the number of cells for the
    grid in the longitude (E-W) direction at level i,
    cell j.
    '''
    if i == 1 and j == 1:
        # use a fixed low resolution for
        # the first level grid
        m_phi = 20
    else:
        m_phi = l_phi * _grid_build_coef(k,
                                         l_lambda,
                                         l_phi,
                                         N)

    return m_phi

def _grid_build_coef(k=1.0, l_lambda=1, l_phi=1, N=0):
    # utility function needed by calc_m_lambda and calc_m_phi
    return np.sqrt((k * N) / (l_lambda * l_phi))

def inclusion_property(center_1_property,
                       intersection_count):
    '''
    Determine the spherical polygon inclusion property
    (inside / outside of polygon) of center 2 based
    on the known polygon inclusion property of center 1
    and the number of spherical polygon edges crossed by
    the minor arc connecting center 2 back to center 1.

    Both center points are the centers of grid cells.

    Based on Table 1 in the manuscript

    center_1_property: the known spherical polygon inclusion
    property of grid center point 1; should be either
    'inside' or 'outside'

    intersection_count: the number of spherical polygon
    edges intersected when connecting the minor arc
    between grid center 1 and grid center 2

    Returns: the spherical polygon inclusion property
    of grid center point 2
    '''
    if center_1_property == 'inside':
        if intersection_count % 2 != 0:
            return 'outside'
        else:
            return 'inside'
    else:
        if intersection_count % 2 != 0:
            return 'inside'
        else:
            return 'outside'

def arc_plane_side(center,
                   point_A,
                   point_B,
                   point_C,
                   ):
    '''
    Determine which side of a great circle plane
    a point on the surface of the sphere is on. The
    anticipated use case is as part of the process
    for determing if two minor arcs intersect (i.e.,
    if the arc connecting two center points in grid
    intersects with an arc representing an edge
    of the spherical polygon). In that case,
    this function would likely be called four
    times--all four points must be on opposite sides
    of the opposite arc.

    Calculations are performed in rectangular (Cartesian)
    coordinate system, as described in manuscript.

    center: the center of the sphere on which
            the point in spherical polygon test
            is being performed [x,y,z]

    point_A: the Cartesian coords of point A

    point_B: the Cartesian coords of point B

    point_C: Cartesian coords of point C,
             for which it is desired to assess
             which side of the plane through
             A, B & center it lies on

    returns w: > 0 on left side of plane;
               < 0 on right side of plane
    '''
    w = np.dot(np.cross((point_A - center),
                        (point_B - point_A)),
               point_C - point_A)
    return w

def determine_arc_intersection(point_A,
                               point_B,
                               point_C,
                               point_D,
                               center):
    '''
    Determine if two spherical
    arcs intersect.

    The conditions described in the
    paragraph at the top of page 61
    in the manuscript must be satisfied
    for two spherical arcs to be labelled
    as intersecting.

    In short, points AB must be on opposite
    sides of the plane formed by CD and
    the center of the sphere, and the reverse
    must also be true.

    point_A: The Cartesian coord of the
             start of spherical arc 1

    point_B: The Cartesian coord of the
             end of spherical arc 1

    point_C: The Cartesian coord of the
             start of spherical arc 2

    point_D: The Cartesian coord of the
             end of spherical arc 2

    center: the center of the sphere on which
            the point in spherical polygon test
            is being performed [x,y,z]

    returns bool: True if the arcs
                  intersect; otherwise,
                  False.
    '''
    # TODO: reduce code duplication
    # by abstracting some repeated
    # logic here
    w_vals = []
    for point in [point_A, point_B]:
        w = arc_plane_side(center,
                           point_C,
                           point_D,
                           point)
        w_vals.append(w)

    if np.sign(w_vals).sum() != 0:
        # same side of opposite plane
        # so no intersection of arcs
        return False

    w_vals = []
    for point in [point_C, point_D]:
        w = arc_plane_side(center,
                           point_A,
                           point_B,
                           point)
        w_vals.append(w)

    if np.sign(w_vals).sum() != 0:
        # same side of opposite plane
        # so no intersection of arcs
        return False

    return True

def cast_grid_level_1():
    '''
    Cast the initial (level 1)
    grid on the sphere. As described
    in the manuscript, the level 1 grid
    has fixed size / coarse resolution.

    Returns: mesh-grid ndarrays at
             the fixed dimensionalities
             specified by Equation 1
             in manuscript
    '''

    # the first level grid is at
    # a fixed resolution that is
    # coarse & uniform
    m_lambda_1 = calc_m_lambda(i=1, j=1)
    m_phi_1 = calc_m_phi(i=1, j=1)

    level_1 = np.mgrid[-np.pi / 2.:np.pi / 2.:complex(m_lambda_1),
                       -np.pi:np.pi:complex(m_phi_1)]
    # level_1 grid should have shape:
    # (2, m_lambda_1, m_phi_1)
    return level_1

# NOTE: this function is in
# the very early stages of dev
def cast_subgrids(spherical_polyon,
                  MAXD=4):
    '''
    spherical_polyon: an ndarray of the
                      sorted vertices of
                      the spherical polygon
                      in Cartesian coords
                      shape -- (N, 3)

    MAXD is the set maximum level of the
    multilevel grids

    '''

    # divide the initial fixed uniform
    # grid on the sphere into uniform
    # subgrids based on spherical
    # polygon edge containment

    level_1 = cast_grid_level_1()

    # Cartesian coordinates are
    # used to determine arc intersections
    # so it will be necessary to probe
    # the interactions between
    # spherical polygon edges & grid
    # cells accordingly


    # I will need an appropriate data
    # structure for storing the values
    # of N for each grid cell in level 1,
    # where N is the number of spherical
    # polygon edges determined to be in
    # the grid cell

    # level_1 has data structure with shape:
    # (2, n_latitude, n_longitude)
    # each row in the first array is just
    # a duplication of the same lat value
    # i.e., [0, 0, 0, ...], [1, 1, 1, ...]
    # each row in the second array cycles
    # through all possible long values
    # i.e., [0, 1, 2, ...], [0, 1, 2, ...]

    level_1_lat = level_1[0]
    level_1_long = level_1[1]

    # a grid cell has four vertices and
    # four edges
    # these can be assigned / indexed as: i, i + 1
    # for matching rows from lat and lon
    # arrays [i.e., rows 0 and 1 with i, i + 1
    # for both lat and lon]

    # it will likely be necessary to iterate by just
    # +1 each time since the right edge of one grid
    # cell is also the left edge of the next, etc.
    # likewise for tops & bottoms of grid cells

    for i in range(level_1_lat.shape[0]):
        # grid cell vertex coords (lambda, phi)
        # or (latitude, longitude):
        for j in range(level_1_long[0].size):
            top_left_corner = np.array([level_1_lat[i][j],
                                        level_1_long[i][j]])
            top_right_corner = np.array([level_1_lat[i][j + 1],
                                        level_1_long[i][j + 1]])
            bottom_left_corner = np.array([level_1_lat[i + 1][j],
                                           level_1_long[i + 1][j]])
            bottom_right_corner = np.array([level_1_lat[i + 1][j + 1],
                                           level_1_long[i + 1][j + 1]])
