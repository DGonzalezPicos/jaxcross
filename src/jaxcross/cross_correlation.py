# import numpy as np

from scipy.interpolate import splev, splrep
c = 2.998e5 # km/s

import jax.numpy as np
from jax import vmap, devices, jit
from functools import wraps
import time
from jaxcross import Template


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} --- {total_time*1e3:.2f} ms')
        return result
    return timeit_wrapper

    
med_sub = lambda x: np.subtract(x, np.median(x))
class CCF:
    def __init__(self, RV, model):
        self.RV = RV
        self.model = model
        self.beta = 1 - (self.RV/c) 
        self.cs = splrep(self.model.x, med_sub(self.model.y))
                
                
    def shift_template(self, datax):
        self.g = splev(np.outer(datax,self.beta), self.cs)
        return self
     
    
    @timeit
    def xcorr(self, f):
      return np.dot(med_sub(f) / np.var(f, axis=0), self.g)
    
    
    def __call__(self, datax, datay, jit_enable=True):
        self.shift_template(datax)
        if jit_enable:
            jit_ccf = jit(self.xcorr)
            self.map = jit_ccf(datay)
            return self
        self.map = self.xcorr(datay)
        return self
    
    def imshow(self, ax=None, fig=None, title='', **kwargs):
        # plot y-axis as phase (if available)
        if hasattr(self, 'phase'):
            y1, y2 = np.min(self.phase), np.max(self.phase)
        else:
            y1, y2 = 0, self.map.shape[0]-1

        ext = [np.min(self.RV), np.max(self.RV), y1, y2]
        ax = ax or plt.gca()
        obj = ax.imshow(self.map,origin='lower',aspect='auto',
                        extent=ext, **kwargs)
        if not fig is None: fig.colorbar(obj, ax=ax, pad=0.05)

        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='white')
        ax.set(title=title)

        return obj

    @timeit 
    def numpy_ccf(self, datax, datay):
        import numpy
        f = datay - numpy.mean(datay)
        self.shift_template(datax)
        return numpy.dot(f/ np.var(f, axis=0), self.g)
    
if __name__ == '__main__':
    from jax import random, devices
    import matplotlib.pyplot as plt
    import jax
    print(devices())
    # Global flag to set a specific platform, must be used at startup.
    jax.config.update('jax_platform_name', 'cpu') # USE CPU --> not working now
    key = random.PRNGKey(0)
    size = 4000
    mx = np.linspace(0, size*0.1, size)
    my = np.sin(2*mx+0.1)
    x = mx + 0.1*np.min(np.diff(mx))
    
    
    y = np.tile(my, (200,1))
    noise = random.normal(key, (y.shape))
    y += 0.1 * noise
    print('Data size = ', y.shape)

    RV = np.arange(-3000., 3002., 20.)
    print('Template size = ' , RV.shape, my.shape)
    ccf = CCF(RV, Template(mx,my))
    ccf(x,y).imshow()
    plt.show()
    
    compare_numpy = False
    if compare_numpy:
        print('JAX with JIT disabled')
        ccf_map = ccf(x,y, jit_enable=False)
        print('JAX with JIT enabled')
        ccf_map = ccf(x,y, jit_enable=True)
        # print(repr(ccf_map.device_buffer.device()))  # check which devide data is on
        print('Numpy on CPU')
        ccf_map_numpy = ccf.numpy_ccf(x,y)
    
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].imshow(ccf_map)
        ax[1].imshow(ccf_map_numpy)
        plt.show()
    
    