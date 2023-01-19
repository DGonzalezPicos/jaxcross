# import matplotlib.pyplot as plt
from jaxcross.cross_correlation import CCF, Template

import jax.numpy as np
from jax import random, devices
print(devices())
key = random.PRNGKey(0)
size = 5000
mx = np.linspace(0, size*0.1, size)
my = np.sin(2*mx+0.1)
x = mx + 0.1*np.min(np.diff(mx))


noise = random.normal(key, (size,))
y = np.tile(my, (400,1)) + 0.5*noise

RV = np.arange(-1000., 1002., 12.)
ccf = CCF(RV, Template(mx,my))
ccf_map = ccf(x,y)
print('jaxcross has been successfully installed!')

# plt.imshow(ccf_map)
# plt.show()
