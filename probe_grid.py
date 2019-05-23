# rough work to probe / understand / visualize
# the spherical multilevel grid data structures
# produced for a given input spherical polygon
# hopefully, this will help me discover some issues

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import lib
import pyximport; pyximport.install()
from slerp import _slerp as slerp

# the input will be a spherical triangle that
# covers exactly 1/8 the surface area of the unit
# sphere (front right spherical triangle)

spherical_polyon = np.array([[0, 1, 0],
                             [0, 0, 1],
                             [-1, 0, 0]], dtype=np.float64)

# try applying spherical linear interpolation to improve plot
N = spherical_polyon.shape[0]
n_int = 900
interpolated_polygon = np.zeros((N * n_int, 3), dtype=np.float64)
t_values = np.float64(np.linspace(0, 1, n_int))

counter = 0
for i in range(N):
    if i == (N-1):
        next_index = 0
    else:
        next_index = i + 1

    interpolated_polygon[counter:(counter + n_int), ...] = slerp(spherical_polyon[i],
                                                               spherical_polyon[next_index],
                                                               n_int,
							       t_values)
    counter += n_int

results = lib.cast_subgrids(spherical_polyon=spherical_polyon,
                            MAXD=4)

(edge_count_array_L1,
cartesian_coords_cells_L1) = results

# plot the level 1 grid on the unit sphere
# along with the spherical polygon, albeit with
# crude matplotlib 3D handling
fig_level_1 = plt.figure()
ax = fig_level_1.add_subplot(111, projection='3d')
ax.scatter(cartesian_coords_cells_L1[...,0],
           cartesian_coords_cells_L1[...,1],
           cartesian_coords_cells_L1[...,2],
           marker='.',
           color='black')
polygon = Poly3DCollection([interpolated_polygon], alpha=1.0)
polygon.set_color('purple')
ax.add_collection3d(polygon)
ax.set_aspect('equal')
ax.azim = -30
ax.elev = -30

fig_level_1.savefig("level_1_grid.png", dpi=300)
