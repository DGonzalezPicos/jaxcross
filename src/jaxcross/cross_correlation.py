# import numpy as np

from scipy.interpolate import splev, splrep, interp1d
c = 2.998e5 # km/s

import jax.numpy as np
from jax import vmap, devices, jit
from functools import wraps
import time
import matplotlib.pyplot as plt
import copy


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
        # self.cs = splrep(self.model.wave, med_sub(self.model.flux))
        self.inter = interp1d(self.model.wave, med_sub(self.model.flux), kind='linear')
                
                
    def shift_template(self, datax):
        # self.g = splev(np.outer(datax,self.beta), self.cs)
        self.nans = np.isnan(datax)
        self.g = self.inter(np.outer(datax[~self.nans],self.beta))
        print(self.g.shape)
        return self
     
    
    @timeit
    def xcorr(self, f):
    #   return np.dot(med_sub(f) / np.var(f, axis=0), self.g)
        return np.dot(f, self.g)
    
    
    
    def __call__(self, datax, datay, jit_enable=True):
        self.shift_template(datax)
        if jit_enable:
            jit_ccf = jit(self.xcorr)
            self.map = jit_ccf(datay[:,~self.nans])
            return self
        self.map = self.xcorr(datay[:,~self.nans])
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
    
    def copy(self):
        return copy.deepcopy(self)

    @timeit 
    def numpy_ccf(self, datax, datay):
        import numpy
        f = datay - numpy.mean(datay)
        self.shift_template(datax)
        return numpy.dot(f/ np.var(f, axis=0), self.g)
    
class KpV:
    def __init__(self, ccf=None, planet=None, deltaRV=None,
                 kp_radius=50., vrest_max=80., bkg=None):
        if not ccf is None:
            self.ccf = ccf.copy()
            self.planet = copy.deepcopy(planet)
            self.dRV = deltaRV or ccf.dRV

            self.kpVec = self.planet.Kp + np.arange(-kp_radius, kp_radius, self.dRV)
            self.vrestVec = np.arange(-vrest_max, vrest_max+self.dRV, self.dRV)
            self.bkg = bkg or vrest_max*0.60

            try:
                self.planet.frame = self.ccf.frame
                # print(self.planet.frame)
            except:
                print('Define data rest frame...')
            # self.n_jobs = 6 # for the functions that allow parallelisation

    def shift_vsys(self, iObs):
        print(iObs)
        outRV = self.vrestVec + self.rv_planet[iObs]
        return interp1d(self.ccf.RV, self.ccf.map[iObs,])(outRV)
    @property
    def snr(self):
        noise_region = np.abs(self.vrestVec)>self.bkg
        noise = np.std(self.ccf_map[:,noise_region])
        bkg = np.median(self.ccf_map[:,noise_region])
        return((self.ccf_map - bkg) / noise)

    @property
    def noise(self):
        '''
        Return the standard deviation of the region away from the peak i.e.
        KpV.vrestVec > KpV.bkg
        '''
        noise_region = np.abs(self.vrestVec)>self.bkg
        return np.std(self.ccf_map[:,noise_region])
    @property
    def baseline(self):
        '''
        Return the median value away from the peak i.e.
        KpV.vrestVec > KpV.bkg
        '''
        noise_region = np.abs(self.vrestVec)>self.bkg
        return np.median(self.ccf_map[:,noise_region])

    def run(self, ignore_eclipse=True, ax=None):
        '''Generate a Kp-Vsys map
        if snr = True, the returned values are SNR (background sub and normalised)
        else = map values'''

        ecl = False * np.ones_like(self.planet.RV)
        if ignore_eclipse:
            ecl = self.planet.mask_eclipse(return_mask=True)

        ccf_map = np.zeros((len(self.kpVec), len(self.vrestVec)))

        for ikp in range(len(self.kpVec)):
            self.planet.Kp = self.kpVec[ikp]
            pRV = self.planet.RV
            for iObs in np.where(ecl==False)[0]:
                outRV = self.vrestVec + pRV[iObs]
                ccf_map[ikp,] += interp1d(self.ccf.RV, self.ccf.map[iObs,])(outRV)
        self.ccf_map = ccf_map

        # self.bestSNR = self.snr.max() # store info as variable
        if ax != None: self.imshow(ax=ax)
        return self
    
if __name__ == '__main__':
    from jax import random, devices
    import matplotlib.pyplot as plt
    import jax
    from jaxcross import Template

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
    
    