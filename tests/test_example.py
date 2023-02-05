# import matplotlib.pyplot as plt
from jaxcross import CCF, Template

import jax.numpy as np
from jax import random, devices

import matplotlib.pyplot as plt
print(devices())
key = random.PRNGKey(0)
size = 5000
mx = np.linspace(0, size*0.1, size)
my = np.sin(100*mx+0.1)
x = mx + 0.1*np.min(np.diff(mx))

dRV = 40. # shift in km/s to synthetic data relative to template
c = 299792.458 # speed of light in km/s
x = mx * (1 + dRV/c)


noise = random.normal(key, (size,))
y = np.tile(my, (200,1)) + 0.5*noise

RV = np.arange(-200., 202., 0.5)
ccf = CCF(RV, Template(mx,my), interpolation='cubic')
ccf_map = ccf(x,y)

fig, ax = plt.subplots(2,1, figsize=(12,4), sharex=True)
fig.subplots_adjust(hspace=0.01)
ccf_map.imshow(ax=ax[0])
ax[1].plot(RV, np.median(ccf_map.map, axis=0))
[axi.axvline(dRV, color='r', linestyle='--', label=f'RV={dRV} km/s') for axi in ax]
ax[0].set_ylabel('Frame number')
ax[1].set(ylabel='CCF', xlabel='RV [km/s]')
ax[1].legend()
plt.show()
print('jaxcross has been successfully installed!')

# plt.imshow(ccf_map)
# plt.show()
