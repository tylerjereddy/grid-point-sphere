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
import copy
import scipy
import scipy.spatial

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

# color code cells by amount of spherical
# polygon edges contained
# here I just happen to know the max is 2
colors = {0: 'black', 1: 'orange', 2: 'red'}
# we don't want to plot over edges already plotted
# with a higher containment count, so keep track of this
dict_edge_data = {}
counter = 0
for cell, edge_count in zip(cartesian_coords_cells_L1, edge_count_array_L1):
    # parse all four edges of the cell
    cycle_cell = np.empty((5, 3))
    cycle_cell[:4, ...] = cell
    cycle_cell[4, ...] = cell[0, ...]
    for i in range(4):
        edge = cycle_cell[i:i+2]
        dict_edge_data[counter] = {'edge': edge,
                                   'edge_count': edge_count}
        counter += 1

# now move through dict_edge_data and plot edges using
# color that matches higher spherical polygon edge containment
# count only
internal_dict = copy.deepcopy(dict_edge_data)
iter_count = 0
total_iter = len(dict_edge_data)
plot = True

for key, edge_entry in dict_edge_data.items():
    current_edge = edge_entry['edge']
    current_edge_count = edge_entry['edge_count']
    dist = scipy.spatial.distance.cdist(spherical_polyon,
                                        current_edge).min()
    if current_edge_count > 0:
        msg = ("dist violation for current_edge_count: " + str(current_edge_count) +
                "; edge: " + str(current_edge) +
                "; distance: " + str(dist))
        assert dist <= np.sqrt(2), msg

    for subkey, subentry in internal_dict.items():
        reference_edge = subentry['edge']
        reference_count = subentry['edge_count']
        if (np.allclose(current_edge, reference_edge) or
            np.allclose(current_edge, reference_edge[::-1])):
            if current_edge_count < reference_count:
                plot = False
                break
    if plot:
        ax.plot(current_edge[..., 0],
                current_edge[..., 1],
                current_edge[..., 2],
                color=colors[current_edge_count])
    plot = True
    iter_count += 1
    print(iter_count, 'of', total_iter, 'iterations')

polygon = Poly3DCollection([interpolated_polygon], alpha=0.3)
polygon.set_color('purple')
ax.add_collection3d(polygon)
ax.azim = -30
ax.elev = -30

fig_level_1.savefig("level_1_grid.png", dpi=300)
