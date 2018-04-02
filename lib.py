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
    w = np.dot(np.cross(np.stack([center, point_A]),
                        np.stack([point_A, point_B])),
               np.stack([point_A, point_C]))
    return w
