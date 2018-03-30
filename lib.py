'''Library of utility functions for the point in spherical polygon
algorithm implementation described by Li et al. (2017).
'''

import numpy as np

def calc_m_lambda(i, j):
    '''
    From equation 1 in Li et al. (2017)

    i is the level of the grid

    j is the grid on the level designated by i

    Returns m_lambda -- the number of cells for the
    grid in the latitude (N-S) direction.
    '''
    if i == 1 and j == 1:
        # use a fixed low resolution for
        # the first level grid
        m_lambda = 10

    # TODO: implement deeper levels of the grid

    return m_lambda

def calc_m_phi(i, j):
    '''
    From equation 1 in Li et al. (2017)

    i is the level of the grid

    j is the grid on the level designated by i

    Returns m_phi -- the number of cells for the
    grid in the longitude (E-W) direction.
    '''
    if i == 1 and j == 1:
        # use a fixed low resolution for
        # the first level grid
        m_phi = 20

    # TODO: implement deeper levels of the grid

    return m_phi
