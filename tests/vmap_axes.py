from jax import vmap, jit, grad
import jax.numpy as np

# a = np.vstack(np.arange(3) for _ in range(4))
# b = np.tile(np.arange(3), (4,1))
# # a and b are identical although they are initialised differently

# def add(x, y):
#     return x + y

# c000 = vmap(add, in_axes=(0,0), out_axes=0)(a, b)
# c001 = vmap(add, in_axes=(0,0), out_axes=1)(a, b) # c001 = c000.T
# c110 = vmap(add, in_axes=(1,1), out_axes=0)(a, b) # c110 = c001
# c111 = vmap(add, in_axes=(1,1), out_axes=1)(a, b) # c100 = c000

c = np.tile(np.arange(3), (3,1)) # size (3,3)
d = np.tile(np.arange(3,8), (5,1)) # size (5,5)

def prod(x,y):
    return np.outer(x,y)
# p0N0 = vmap(prod, in_axes=(0, None), out_axes=0)(c, d)
# pN00 = vmap(prod, in_axes=(None, 0), out_axes=0)(c, d)

p = jit(vmap(prod, in_axes=(0, None), out_axes=0))(c,d)