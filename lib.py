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

    # TODO: implement deeper levels of the grid

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

    # TODO: implement deeper levels of the grid

    return m_phi
