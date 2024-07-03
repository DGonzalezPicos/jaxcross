import jax.numpy as np
from jax import random
import numpy


key = random.PRNGKey(0)
A = random.normal(key, (100, 7, 3, 2048))
cube = A[:,0,0,:]
mask = cube[0,] < 0.01
masked_cube = cube[:,~mask]
# B = random.normal(key, (100, 2028)) + 10.

u, s, vh = numpy.linalg.svd(masked_cube, full_matrices=False, compute_uv=True)
N = 4
s[0:N] = 0.
W=numpy.diag(s)
rec_cube = numpy.dot(u,numpy.dot(W,vh))
new_cube = cube.at[:,~mask].set(rec_cube)
# C = A.at[:,0,0,:].set(new_B) 
