'''Library of utility functions for the point in spherical polygon
algorithm implementation described by Li et al. (2017).
'''

import math
import numpy as np
from numpy.testing import assert_allclose


def generate_level_subgrids(dict_level_n,
                            grid_cell_counter_previous_level,
                            edge_count_array_previous_level,
                            target_level,
                            l_lambda,
                            l_phi,
                            level_n_lat,
                            level_n_long):
    '''
    For all levels below level 1,
    the number of spherical polygon
    edges contained within grid cells
    (along with a few other parameters)
    is used to determine the degree
    of subdivision into smaller cells.
    The generation of the set of subgrids
    at each level below 1 is handled
    by this function.

    Currently modifies a dict
    where each key, value pair
    represents one of the subgrids
    at the given dict level.
    '''
    lambda_expansions = np.zeros(edge_count_array_previous_level.size)
    phi_expansions = np.zeros(edge_count_array_previous_level.size)

    for grid_index in range(grid_cell_counter_previous_level):
        N = edge_count_array_previous_level[grid_index]
        lambda_expansions[grid_index] = calc_m_lambda(i=target_level,
                                                      j=1,  # not used really
                                                      l_lambda=l_lambda,
                                                      l_phi=l_phi,
                                                      N=N)
        phi_expansions[grid_index] = calc_m_phi(i=target_level,
                                                j=1,  # not used really
                                                l_lambda=l_lambda,
                                                l_phi=l_phi,
                                                N=N)

    for grid_cell in range(grid_cell_counter_previous_level):

        m_lambda = lambda_expansions[grid_cell]
        m_phi = phi_expansions[grid_cell]

        if m_phi and m_lambda:
            # this grid cell contains a spherical polygon edge
            # and is targeted for subdivision into a subdgrid
            grid_key = 'L{level}_grid_num_{num}'.format(num=grid_cell,
                                                        level=target_level)

            # generate a meshgrid within the confined boundaries
            # of the cell at level n

            # first do some awkward mgrid cell retrieval
            # by index striding

            retrieval_counter = 0
            for i in range(level_n_lat.shape[0] - 1):
                for j in range(level_n_long[0].size - 1):
                    if retrieval_counter == grid_cell:
                        # exact identification of the bounds
                        # of subdivision target cell
                        top_lambda_bound = level_n_lat[i][j]
                        bottom_lambda_bound = level_n_lat[i + 1][j]
                        left_phi_bound = level_n_long[i][j]
                        right_phi_bound = level_n_long[i][j + 1]

                        # produce and store the level n grid
                        # to be placed inside the level n-1 cell
                        # that has spherical polygon edges in it

                        # want to include the edges of the original
                        # cell when subidiving new grid so add
                        # in those vals accordingly prior
                        # to grid generation
                        # NOTE: so far, multiplying x 3 seems
                        # to be effective in generating more
                        # reasonable grid structs?
                        m_lambda *= 3
                        m_phi *= 3

                        level_n = np.mgrid[bottom_lambda_bound:
                                           top_lambda_bound:complex(m_lambda),
                                           left_phi_bound:right_phi_bound:
                                           complex(m_phi)]
                        dict_level_n[grid_key] = level_n

                    retrieval_counter += 1


def edge_cross_accounting(level_n_lat,
                          level_n_long,
                          N_edges,
                          grid_cell_edge_counts_level_n,
                          level_n_grid_cell_counter,
                          spherical_polyon):
    '''
    A utility function to handle some of the
    accounting work for assessing whether
    spherical polygon edges are inside
    certain grid cells, and returning
    an appropriate data structure with
    this information.
    '''
    # handy to have Cartesian coords of each
    # cell for plotting
    cart_coords_cells = []
    for i in range(level_n_lat.shape[0] - 1):
        # grid cell vertex coords (lambda, phi)
        # or (latitude, longitude):
        for j in range(level_n_long[0].size - 1):
            top_left_corner = np.array([level_n_lat[i][j],
                                        level_n_long[i][j]])
            top_right_corner = np.array([level_n_lat[i][j + 1],
                                        level_n_long[i][j + 1]])
            bottom_left_corner = np.array([level_n_lat[i + 1][j],
                                           level_n_long[i + 1][j]])
            bottom_right_corner = np.array([level_n_lat[i + 1][j + 1],
                                           level_n_long[i + 1][j + 1]])
            # convert to Cartesian coords
            cart_coords = [top_left_corner,
                           top_right_corner,
                           bottom_right_corner,
                           bottom_left_corner]
            cart_coords_cells.append(cart_coords)

            for k in range(4):
                # hard coding unit radius at the moment
                cart_coords[k] = convert_spherical_array_to_cartesian_array(
                                    np.array([1, cart_coords[k][0],
                                              cart_coords[k][1]]))

            grid_cell_edge_counts_level_n.append(0)
            level_n_grid_cell_counter += 1
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
                for grid_cell_index in [0, 1, 2, 3]:
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
                    # register near-exact matches for lat/long
                    # between grid cell edges and spherical polygon
                    # edges---will aim to mark these grid cells for division
                    edge_overlap = False
                    point_AB_conv = convert_cartesian_to_lat_long(np.array([
                                                                point_A,
                                                                point_B]))
                    point_CD_conv = convert_cartesian_to_lat_long(np.array([
                                                                point_C,
                                                                point_D]))
                    point_A_conv = point_AB_conv[0]
                    point_B_conv = point_AB_conv[1]
                    point_C_conv = point_CD_conv[0]
                    point_D_conv = point_CD_conv[1]

                    # careful at longitude transition point
                    point_A_conv[point_A_conv == -180] = 180
                    point_B_conv[point_B_conv == -180] = 180
                    point_C_conv[point_C_conv == -180] = 180
                    point_D_conv[point_D_conv == -180] = 180
                    point_A_conv[point_A_conv < -180] += 360
                    point_B_conv[point_B_conv < -180] += 360
                    point_C_conv[point_C_conv < -180] += 360
                    point_D_conv[point_D_conv < -180] += 360
                    point_A_conv[point_A_conv > 180] -= 360
                    point_B_conv[point_B_conv > 180] -= 360
                    point_C_conv[point_C_conv > 180] -= 360
                    point_D_conv[point_D_conv > 180] -= 360
                    point_A_conv[abs(point_A_conv) == 360] = 180
                    point_B_conv[abs(point_B_conv) == 360] = 180
                    point_C_conv[abs(point_C_conv) == 360] = 180
                    point_D_conv[abs(point_D_conv) == 360] = 180

                    # if both latitude and longitude change for
                    # a grid cell edge, it is not actually an edge
                    # since it intersects the cell
                    if ((not np.allclose(point_C_conv[0], point_D_conv[0])) and
                        (not np.allclose(point_C_conv[1],
                                         point_D_conv[1]))):
                        edge = str([point_C_conv, point_D_conv])
                        raise ValueError("Self-intersecting grid cell "
                                         "edge detected: " + edge)

                    if point_C_conv[-1] == point_D_conv[-1]:
                        if np.allclose(point_C_conv[-1],
                                       [point_A_conv[-1],
                                        point_B_conv[-1]]):
                            # match on longitude line, but
                            # check for range match---we don't
                            # want to flag grid cells that are out
                            # of latitude range even if parallel to
                            # spherical polygon edge
                            max_val = np.maximum(point_C_conv[0],
                                                 point_D_conv[0])
                            min_val = np.minimum(point_C_conv[0],
                                                 point_D_conv[0])
                            if (max_val >
                                np.minimum(point_A_conv[0], point_B_conv[0])
                                and min_val < np.maximum(point_A_conv[0],
                                                         point_B_conv[0])):
                                edge_overlap = True
                    elif point_C_conv[0] == point_D_conv[0]:
                        if np.allclose(point_C_conv[0],
                                       [point_A_conv[0],
                                        point_B_conv[0]]):
                            # similar range filter check to above
                            max_val = np.maximum(point_C_conv[1],
                                                 point_D_conv[1])
                            min_val = np.minimum(point_C_conv[1],
                                                 point_D_conv[1])
                            if (max_val >
                                np.minimum(point_A_conv[1], point_B_conv[1])
                                and min_val < np.maximum(point_A_conv[1],
                                                         point_B_conv[1])):
                                edge_overlap = True

                    if intersect or edge_overlap:
                        # once a spherical polygon edge is found
                        # to intersect any edge of the grid cell
                        # we record the presence of that edge
                        # inside the cell & then move on
                        # to the next edge of the spherical polygon

                        # actually, this is too generous---it can record
                        # plane intersections when the arcs are really nowhere
                        # near each other on the sphere surface due to
                        # extrapolation;
                        # try to filter at least crudely using coordinate
                        # limits

                        true_intersection = 1
                        for coord_axis in range(3):
                            if (np.minimum(point_A, point_B)[coord_axis] >
                               np.maximum(point_C, point_D)[coord_axis]):
                                # there's no way spherical polygon edge AB
                                # intersects CD grid edge because AB is above
                                # CD on the sphere X,Y, or Z coord
                                true_intersection = 0
                                break
                            elif (np.maximum(point_A, point_B)[coord_axis] <
                                  np.minimum(point_C, point_D)[coord_axis]):
                                # same argument with AB completely below
                                # CD
                                true_intersection = 0
                                break

                        if true_intersection or edge_overlap:
                            grid_cell_edge_counts_level_n[
                                        level_n_grid_cell_counter - 1] += 1
                        break

    return (grid_cell_edge_counts_level_n, level_n_grid_cell_counter,
            np.array(cart_coords_cells))


def convert_spherical_array_to_cartesian_array(spherical_coord_array,
                                               angle_measure='radians'):
    '''
    Take shape (N,3) spherical_coord_array (r,theta,phi)
    and return an array of the same shape in cartesian
    coordinate form (x,y,z). Based on the equations
    provided at:
    http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations
    #From_spherical_coordinates
    use radians for the angles by default,
    degrees if angle_measure == 'degrees'
    '''
    cartesian_coord_array = np.zeros(spherical_coord_array.shape)
    # convert to radians if degrees are used in input
    # (prior to Cartesian conversion process)
    if angle_measure == 'degrees':
        spherical_coord_array[..., 1] = np.deg2rad(
                                            spherical_coord_array[..., 1])
        spherical_coord_array[..., 2] = np.deg2rad(
                                            spherical_coord_array[..., 2])
    # now the conversion to Cartesian coords
    cartesian_coord_array[..., 0] = (spherical_coord_array[..., 0] *
                                     np.cos(spherical_coord_array[..., 1]) *
                                     np.cos(spherical_coord_array[..., 2]))
    cartesian_coord_array[..., 1] = (spherical_coord_array[..., 0] *
                                     np.cos(spherical_coord_array[..., 1]) *
                                     np.sin(spherical_coord_array[..., 2]))
    cartesian_coord_array[..., 2] = (spherical_coord_array[..., 0] *
                                     np.sin(spherical_coord_array[..., 1]))
    return cartesian_coord_array


def convert_cartesian_to_lat_long(cartesian_coord_array):
    # https://gis.stackexchange.com/a/120740
    output_array = np.empty(cartesian_coord_array.shape)[..., :-1]
    # assume unit sphere:
    r = 1.0
    # latitude:
    output_array[..., 0] = (np.arcsin(cartesian_coord_array[..., 2]/r) *
                            180. / np.pi)
    # longitude:
    output_array[..., 1] = (np.arctan2(cartesian_coord_array[..., 1],
                                       cartesian_coord_array[..., 0]) *
                            180. / np.pi)

    output_array[..., 1][(cartesian_coord_array[..., 0] <= 0) &
                         (cartesian_coord_array[..., 1] > 0)] += 180
    output_array[..., 1][(cartesian_coord_array[..., 0] <= 0) &
                         (cartesian_coord_array[..., 1] <= 0)] -= 180

    # the longitude value of a N/S pole coordinate is ambiguous,
    # so in that case just use the same longitude value as the other
    # coordinate in pair
    if np.allclose(cartesian_coord_array[0], np.array([0, 0, 1])):
        # +90 latitude at North Pole
        output_array[0, 0] = 90.0
        # same longitude as other coord:
        output_array[0, 1] = output_array[1, 1]
    elif np.allclose(cartesian_coord_array[0], np.array([0, 0, -1])):
        # -90 latitude at South Pole
        output_array[0, 0] = -90.0
        # same longitude as other coord:
        output_array[0, 1] = output_array[1, 1]
    elif np.allclose(cartesian_coord_array[1], np.array([0, 0, -1])):
        # -90 latitude at South Pole
        output_array[1, 0] = -90.0
        # same longitude as other coord:
        output_array[1, 1] = output_array[0, 1]
    elif np.allclose(cartesian_coord_array[1], np.array([0, 0, 1])):
        # 90 latitude at North Pole
        output_array[1, 0] = 90.0
        # same longitude as other coord:
        output_array[1, 1] = output_array[0, 1]
    return output_array


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

    return math.ceil(m_lambda)


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

    return math.ceil(m_phi)


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
    for determining if two minor arcs intersect (i.e.,
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
    # TODO: if points A and B are antipodes
    # they define an infinite set of great
    # circle planes, so we may need error
    # handling
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
             the fixed dimensions
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

    (grid_cell_edge_counts_level_1,
     grid_cell_counter,
     cart_coords_L1) = edge_cross_accounting(level_1_lat,
                                             level_1_long,
                                             N_edges,
                                             grid_cell_edge_counts_level_1,
                                             grid_cell_counter,
                                             spherical_polyon)

    edge_count_array = np.array(grid_cell_edge_counts_level_1)

    # draft code to determine the number of lambda (N-S latitude)
    # divisions needed for each grid cell in level 1 based
    # on the number of spherical polygon edges that each contains

    # let's just use crude Python looping for now & worry
    # about vectorization later
    lambda_expansions = np.zeros(edge_count_array.size)
    # similar data structure for longitude (E-W) expansion
    # accounting
    phi_expansions = np.zeros(edge_count_array.size)

    # l_lambda should be number of degrees of latitude spanned by
    # a given grid cell at level 1
    l_lambda = np.pi / 10.
    # similarly for longitude at level 1
    l_phi = (2 * np.pi) / 20.

    # we're actually targeting level 2 (which is inside level 1)
    # eventually will iterate through target_level programmatically
    # but manual for current stage of dev

    dict_level_2 = {}
    generate_level_subgrids(dict_level_n=dict_level_2,
                            grid_cell_counter_previous_level=grid_cell_counter,
                            edge_count_array_previous_level=edge_count_array,
                            target_level=2,
                            l_lambda=l_lambda,
                            l_phi=l_phi,
                            level_n_lat=level_1_lat,
                            level_n_long=level_1_long)

    # start processing level 2 grid data (should eventually
    # be able to reduce code duplication & combine levels
    # in a loop)

    grid_cell_edge_counts_level_2 = []
    L2_grid_cell_counter = 0
    cart_coords_L2 = []

    for level_2_grid_key in sorted(dict_level_2.keys()):
        # level 2 has many grids
        # so we iterate through the
        # cells of each of those grids
        level_2_grid = dict_level_2[level_2_grid_key]

        # now generate the level_x_lat and
        # level_x_long vars like we did with level
        # 1 previously
        level_2_lat = level_2_grid[0]
        level_2_long = level_2_grid[1]

        # try looping over these values in
        # all the subgrids (but be careful not
        # to reset values between L2 subgrids)
        (grid_cell_edge_counts_level_2,
         L2_grid_cell_counter,
         cart_coords_L2_tmp) = edge_cross_accounting(
                                                 level_2_lat,
                                                 level_2_long,
                                                 N_edges,
                                                 grid_cell_edge_counts_level_2,
                                                 L2_grid_cell_counter,
                                                 spherical_polyon)
        cart_coords_L2.append(cart_coords_L2_tmp)
    # now we have the data structure containing
    # the number of spherical polygon edges
    # contained within each L2 grid cell
    grid_cell_edge_counts_level_2 = np.array(grid_cell_edge_counts_level_2)

    # level 2 grid should be subdivided with more grid cells containing edges:
    assert grid_cell_edge_counts_level_2.sum() > edge_count_array.sum()

    # produce level 3 grid data structure
    # here we have to loop through each of the level 2 grids
    # so it is more involved than generating level 2 from the fixed
    # level 1
    dict_level_3 = {}
    grid_index = 0
    for subgrid_num, key in enumerate(sorted(dict_level_2.keys())):
        sub_key = "level_3_subgrid_{num}".format(num=subgrid_num)
        dict_level_3[sub_key] = {}
        level_2_grid = dict_level_2[key]
        level_2_lat = level_2_grid[0]
        level_2_long = level_2_grid[1]

        # iterate over N cells in subgrid
        N = (level_2_lat.shape[1] - 1) * (level_2_long.shape[1] - 1)

        # TODO: level-appropriate values of l_lambda / l_phi ??
        k = grid_cell_edge_counts_level_2[grid_index:grid_index + N]
        generate_level_subgrids(dict_level_n=dict_level_3[sub_key],
                                grid_cell_counter_previous_level=N,
                                edge_count_array_previous_level=k,
                                target_level=3,
                                l_lambda=l_lambda,
                                l_phi=l_phi,
                                level_n_lat=level_2_lat,
                                level_n_long=level_2_long)
        grid_index += N

    # start processing level 3 grid data (should eventually
    # be able to reduce code duplication & combine levels
    # in a loop)

    grid_cell_edge_counts_level_3 = []
    L3_grid_cell_counter = 0
    cart_coords_L3 = []

    for level_3_grid_key in sorted(dict_level_3.keys()):
        # level 3 has many grids
        # so we iterate through the
        # cells of each of those grids
        level_3_grid = dict_level_3[level_3_grid_key]
        for key, value in level_3_grid.items():
            if len(value) > 0:

                # now generate the level_x_lat and
                # level_x_long vars like we did with level
                # 1 previously
                level_3_lat = value[0]
                level_3_long = value[1]

                # try looping over these values in
                # all the subgrids (but be careful not
                # to reset values between L3 subgrids)
                (grid_cell_edge_counts_level_3,
                 L3_grid_cell_counter,
                 cart_coords_L3_tmp) = edge_cross_accounting(
                                        level_3_lat,
                                        level_3_long,
                                        N_edges,
                                        grid_cell_edge_counts_level_3,
                                        L3_grid_cell_counter,
                                        spherical_polyon)
                cart_coords_L3.append(cart_coords_L3_tmp)

    # now we have the data structure containing
    # the number of spherical polygon edges
    # contained within each L3 grid cell
    grid_cell_edge_counts_level_3 = np.array(grid_cell_edge_counts_level_3)

    # for the subdivision algorithm, each subsequent level should have a higher
    # count of grid cells containing spherical polygon edges
    assert (grid_cell_edge_counts_level_3.sum() >
            grid_cell_edge_counts_level_2.sum())

    # produce level 4 grid data structure
    # here we have to loop through each of the level 3 grids
    dict_level_4 = {}
    grid_index = 0
    for subgrid_num, key in enumerate(sorted(dict_level_3.keys())):
        sub_key = "level_4_subgrid_{num}".format(num=subgrid_num)
        dict_level_4[sub_key] = {}
        level_3_grid = dict_level_3[key]
        for key, value in level_3_grid.items():
            if len(value) > 0:
                level_3_lat = value[0]
                level_3_long = value[1]

                # iterate over N cells in subgrid
                N = (level_3_lat.shape[1] - 1) * (level_3_long.shape[1] - 1)

                # TODO: level-appropriate values of l_lambda / l_phi ??
                k = grid_cell_edge_counts_level_3[grid_index:grid_index + N]
                generate_level_subgrids(dict_level_n=dict_level_4[sub_key],
                                        grid_cell_counter_previous_level=N,
                                        edge_count_array_previous_level=k,
                                        target_level=4,
                                        l_lambda=l_lambda,
                                        l_phi=l_phi,
                                        level_n_lat=level_3_lat,
                                        level_n_long=level_3_long)
                grid_index += N

    # start processing level 4 grid data (should eventually
    # be able to reduce code duplication & combine levels
    # in a loop)

    grid_cell_edge_counts_level_4 = []
    L4_grid_cell_counter = 0
    cart_coords_L4 = []

    for level_4_grid_key in sorted(dict_level_4.keys()):
        # level 4 has many grids
        # so we iterate through the
        # cells of each of those grids
        level_4_grid = dict_level_4[level_4_grid_key]
        for key, value in level_4_grid.items():
            if len(value) > 0:

                # now generate the level_x_lat and
                # level_x_long vars like we did with level
                # 1 previously
                level_4_lat = value[0]
                level_4_long = value[1]

                # try looping over these values in
                # all the subgrids (but be careful not
                # to reset values between L4 subgrids)
                (grid_cell_edge_counts_level_4,
                    L4_grid_cell_counter,
                    cart_coords_L4_tmp) = edge_cross_accounting(
                                                level_4_lat,
                                                level_4_long,
                                                N_edges,
                                                grid_cell_edge_counts_level_4,
                                                L4_grid_cell_counter,
                                                spherical_polyon)
                cart_coords_L4.append(cart_coords_L4_tmp)

    # now we have the data structure containing
    # the number of spherical polygon edges
    # contained within each L4 grid cell
    grid_cell_edge_counts_level_4 = np.array(grid_cell_edge_counts_level_4)

    # for the subdivision algorithm, each subsequent level should have a higher
    # count of grid cells containing spherical polygon edges
    assert (grid_cell_edge_counts_level_4.sum() >
            grid_cell_edge_counts_level_3.sum())

    # NOTE: this isn't likely what I'll want to return
    # in final version of function;
    # just debugging the first level spherical polygon
    # edge containment assessment within grid
    return (edge_count_array,
            cart_coords_L1,
            grid_cell_edge_counts_level_2,
            np.array(cart_coords_L2),
            grid_cell_edge_counts_level_3,
            np.array(cart_coords_L3),
            grid_cell_edge_counts_level_4,
            np.array(cart_coords_L4))


def grid_center_point(grid_cell_long_1,
                      grid_cell_long_2,
                      grid_cell_lat_1,
                      grid_cell_lat_2):
    '''
    This function should take as input
    the longitude and latitude coordinates
    of a given grid cell.

    This function should return the longitude
    and latitude coordinates of the center
    of the given grid cell (which is used
    downstream in the algorithm described
    in the manuscript).

    As far as I can tell, it should be
    safe to average the longitude and
    latitude values to obtain the center
    coordinate of the grid cell as used
    in the paper, but we should be cautious
    given the spherical geometry.
    '''

    # check input longitude and latitude values
    # raise an appropriate exception if the values
    # fall outside the boundaries specified in the
    # manuscript
    for longi in [grid_cell_long_1, grid_cell_long_2]:
        if longi < -180 or longi > 180:
            raise ValueError

    for lat in [grid_cell_lat_1, grid_cell_lat_2]:
        if lat < -90 or lat > 90:
            raise ValueError

    # handling of antipodal points would be ambiguous so raise
    # appropriate error
    long_diff = grid_cell_long_2 - grid_cell_long_1
    if abs(long_diff) == 180:
        raise ValueError("antipodal points")
    elif abs(long_diff) == 0 and abs(grid_cell_lat_2 - grid_cell_lat_1) == 180:
        raise ValueError("antipodal points")

    center_long = np.average([grid_cell_long_1, grid_cell_long_2])

    # handle straddling over +/- pi boundaries for longitude
    if long_diff < -180:
        center_long += 180

    center_lat = np.average([grid_cell_lat_1, grid_cell_lat_2])
    return np.array([center_lat, center_long])


def determine_first_traversal_point(first_cell_lat_1,
                                    first_cell_lat_2,
                                    first_cell_long_1,
                                    first_cell_long_2,
                                    list_edges_in_first_cell,
                                    center,
                                    radius):
    '''
    Determine if the first grid cell center
    point on a traversal path is inside
    or outside the spherical polygon.

    This function aims to implement the
    algorithm depicted in Figure 4
    of the manuscript (and described
    in the paragraph above that Figure).

    The first grid cell on a traversal
    path MUST contain at least one edge
    of the spherical polygon.

    Return the string 'inside' if the center
    of the first grid cell is inside the
    spherical polygon; otherwise, return
    the string 'outside.'

    NOTE: this appears to be limited to working
    for the 'bottom left' grid cell that contains
    a spherical polygon edge -- so, we likely
    want to start with a grid cell that has
    minimized latitude and perhaps West longitude

    NOTE 2: actually, looks like we just have to be
    very careful to discern CCW vs. CW ordering on
    list_edges_in_first_cell from the polygon
    '''
    # determine the center (centroid) of the first
    # grid cell on the traversal path, which the
    # manuscript describes as O_i
    first_cell_avg_lat = np.average([first_cell_lat_1,
                                     first_cell_lat_2])
    first_cell_avg_long = np.average([first_cell_long_1,
                                      first_cell_long_2])
    O_i = np.array([[radius,
                    first_cell_avg_lat,
                    first_cell_avg_long]])

    O_i_Cart = convert_spherical_array_to_cartesian_array(
                                              O_i.copy(),
                                              angle_measure='degrees')

    # the manuscript defines AB as a spherical polygon
    # edge inside the first cell on the traversal path
    # let's assume that list_edges_in_first_cell is a
    # data structure where each index contains a shape
    # (2, 2) set of spherical coordinates representing
    # one of the spherical polygon edges (arcs) contained within
    # the first cell

    # let's just decide to define AB as the first arc
    # in list_edges_in_first_cell (shouldn't matter
    # which one we pick if there are > 1)
    point_A = np.array([radius,
                        list_edges_in_first_cell[0][0][0],
                        list_edges_in_first_cell[0][0][1]])

    point_B = np.array([radius,
                        list_edges_in_first_cell[0][1][0],
                        list_edges_in_first_cell[0][1][1]])

    point_A_Cart = convert_spherical_array_to_cartesian_array(
                                                 point_A.copy(),
                                                 angle_measure='degrees')
    point_B_Cart = convert_spherical_array_to_cartesian_array(
                                                 point_B.copy(),
                                                 angle_measure='degrees')

    # the manuscript describes C as the midpoint of AB
    point_C = np.average(list_edges_in_first_cell[0], axis=0)

    # the manuscript describes O_iC as the arc segment
    # connecting O_i and C
    O_iC = np.concatenate((O_i,
                           np.array([[radius, point_C[0], point_C[1]]])))
    O_iC_Cartesian = convert_spherical_array_to_cartesian_array(
                                                    O_iC.copy(),
                                                    angle_measure='degrees')

    # now, we want to count the intersections between
    # O_iC and the spherical polygon edges in the cell
    intersections = 0

    # start from the second edge (if there is one)
    # because AB was from the first edge
    for edge in list_edges_in_first_cell[1:]:
        # need Cartesian coords for
        # determine_arc_intersection() function
        start = np.array([radius, edge[0][0], edge[0][1]])
        end = np.array([radius, edge[1][0], edge[1][1]])
        candidate_edge_start = convert_spherical_array_to_cartesian_array(
                                                     start.copy(),
                                                     angle_measure='degrees')
        candidate_edge_end = convert_spherical_array_to_cartesian_array(
                                                     end.copy(),
                                                     angle_measure='degrees')
        if determine_arc_intersection(point_A=O_iC_Cartesian[0].ravel(),
                                      point_B=O_iC_Cartesian[1].ravel(),
                                      point_C=candidate_edge_start,
                                      point_D=candidate_edge_end,
                                      center=center):
            intersections += 1

    # the algorithm also requires that we determine
    # which side of AB O_i is on
    w = arc_plane_side(center=center,
                       point_A=point_A_Cart.ravel(),
                       point_B=point_B_Cart.ravel(),
                       point_C=O_i_Cart.ravel())

    # finally, determine the inclusion property
    # of O_i and return

    if w > 0:
        # O_i is on left side of AB
        if intersections % 2 == 0:
            # O_i is inside the polygon
            return 'inside'
        else:
            # O_i is outside the polygon
            return 'outside'
    else:
        if intersections % 2 != 0:
            # O_i is inside the polygon
            return 'inside'
        else:
            # O_i is outside the polygon
            return 'outside'
