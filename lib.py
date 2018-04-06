'''Library of utility functions for the point in spherical polygon
algorithm implementation described by Li et al. (2017).
'''

import numpy as np

def convert_spherical_array_to_cartesian_array(spherical_coord_array,angle_measure='radians'):
    '''Take shape (N,3) spherical_coord_array (r,theta,phi) and return an array of the same shape in cartesian coordinate form (x,y,z). Based on the equations provided at: http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
    use radians for the angles by default, degrees if angle_measure == 'degrees' '''
    cartesian_coord_array = np.zeros(spherical_coord_array.shape)
    #convert to radians if degrees are used in input (prior to Cartesian conversion process)
    if angle_measure == 'degrees':
        spherical_coord_array[...,1] = np.deg2rad(spherical_coord_array[...,1])
        spherical_coord_array[...,2] = np.deg2rad(spherical_coord_array[...,2])
    #now the conversion to Cartesian coords
    cartesian_coord_array[...,0] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,1] = spherical_coord_array[...,0] * np.sin(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,2] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,2])
    return cartesian_coord_array

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

    # for now, use a crude list to store the
    # number of spherical polygon edges recorded
    # inside grid cells as we iterate through
    grid_cell_edge_counts_level_1 = []

    # number of edges should match number
    # of vertices in spherical polygon
    N_edges = spherical_polyon.shape[0]

    grid_cell_counter = 0
    for i in range(level_1_lat.shape[0] - 1):
        # grid cell vertex coords (lambda, phi)
        # or (latitude, longitude):
        for j in range(level_1_long[0].size - 1):
            top_left_corner = np.array([level_1_lat[i][j],
                                        level_1_long[i][j]])
            top_right_corner = np.array([level_1_lat[i][j + 1],
                                        level_1_long[i][j + 1]])
            bottom_left_corner = np.array([level_1_lat[i + 1][j],
                                           level_1_long[i + 1][j]])
            bottom_right_corner = np.array([level_1_lat[i + 1][j + 1],
                                           level_1_long[i + 1][j + 1]])
            # convert to Cartesian coords
            cart_coords = [top_left_corner,
                          top_right_corner,
                          bottom_left_corner,
                          bottom_right_corner]

            for k in range(4):
                # hard coding unit radius at the moment
                cart_coords[k] = convert_spherical_array_to_cartesian_array(np.array([1, cart_coords[k][0], cart_coords[k][1]]))

            grid_cell_edge_counts_level_1.append(0)
            grid_cell_counter += 1
            # iterate through the edges (arcs) of the input spherical
            # polygon & record the presence of the edge inside
            # this grid cell in the appropriate data structure
            for edge in range(N_edges):
                current_index = edge
                if current_index == N_edges - 1:
                    next_index = 0
                else:
                    next_index = current_index + 1

                point_A = spherical_polyon[current_index]
                point_B = spherical_polyon[next_index]

                # check for crossing between current
                # spherical polygon arc and the current
                # grid cell edges
                for grid_cell_index in range(4):
                    current_index = grid_cell_index
                    if current_index == 3:
                        next_index = 0
                    else:
                        next_index = current_index + 1

                    point_C = cart_coords[current_index]
                    point_D = cart_coords[next_index]

                    intersect = determine_arc_intersection(point_A,
                                                           point_B,
                                                           point_C,
                                                           point_D,
                                                           np.zeros((3,)))
                    if intersect:
                        # once a spherical polygon edge is found
                        # to intersect any edge of the grid cell
                        # we record the presence of that edge
                        # inside the cell & then move on
                        # to the next edge of the spherical polygon
                        grid_cell_edge_counts_level_1[grid_cell_counter - 1] += 1
                        break

    edge_count_array = np.array(grid_cell_edge_counts_level_1)

    # draft code to determine the number of lambda (N-S latitude)
    # divisions needed for each grid cell in level 1 based
    # on the number of spherical polygon edges that each contains

    # let's just use crude Python looping for now & worry
    # about vectorization later
    lambda_expansions = np.zeros(edge_count_array.size)

    # l_lambda should be number of degrees of latitude spanned by
    # a given grid cell at level 1
    l_lambda = np.pi / 10.
    # similarly for longitude at level 1
    l_phi = ( 2 * np.pi ) / 20.
    for grid_index in range(grid_cell_counter):
        N = edge_count_array[grid_index]
        lambda_expansions[grid_index] = calc_m_lambda(i=1,
                                                      j=1, # not used really
                                                      l_lambda=l_lambda,
                                                      l_phi=l_phi,
                                                      N=N)

    # NOTE: this isn't likely what I'll want to return
    # in final version of function;
    # just debugging the first level spherical polygon
    # edge containment assessment within grid
    return edge_count_array
