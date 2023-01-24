import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import jax
print(jax.devices())


AT=np.random.random((100,500))
B=np.random.random((500,200))
M=AT@B

def svd_numpy(M):
    U, S, V = np.linalg.svd(M, full_matrices=False)
    return U,S,V
    
def svd_jax(M):
    U, S, V = jsp.linalg.svd(M, full_matrices=False)
    return U,S,V

out = svd_numpy(M)
out = svd_jax(M)