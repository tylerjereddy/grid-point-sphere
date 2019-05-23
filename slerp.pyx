import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport (sin, acos, atan2,
                        cos, M_PI, abs)
from libc.stdlib cimport RAND_MAX, rand

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _slerp(double[:] start_coord,
            double[:] end_coord,
            int n_pts,
            double[::1] t_values):
   # spherical linear interpolation between points
   # on great circle arc
   # see: https://en.wikipedia.org/wiki/Slerp#Geometric_Slerp
   # NOTE: could we use scipy.interpolate.RectSphereBivariateSpline instead?
   cdef:
      double omega = acos(np.dot(start_coord, end_coord))
      double sin_omega = sin(omega)
      double[:,:] new_pts = np.empty((n_pts, 3), dtype=np.float64)
      int i, j
      double factors[2]

   for i in xrange(n_pts):
      factors[0] = sin((1 - t_values[i]) * omega) / sin_omega
      factors[1] = sin(t_values[i] * omega) / sin_omega
      for j in range(3):
          new_pts[i,j] = ((factors[0] * start_coord[j]) +
                          (factors[1] * end_coord[j]))
   return new_pts
