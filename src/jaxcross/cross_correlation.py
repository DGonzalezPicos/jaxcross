import numpy

from scipy.interpolate import splev, splrep, interp1d
# from .interpolate import InterpolatedUnivariateSpline ### problems with "jaxlib/gpu/solver_kernels.cc:45" like linalg.svd
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
        units = "s"
        if total_time < 0.1:
            units = "ms"
            total_time *= 1e3
        print(f'Function {func.__name__} --- {total_time:.2f} {units}')
        return result
    return timeit_wrapper

    
med_sub = lambda x: np.subtract(x, np.median(x))
class CCF:
    def __init__(self, RV=None, model=None, interpolation="linear"):
        self.RV = RV
        self.model = model
        if self.RV is not None:
            self.beta = 1 - (self.RV/c) 
        # self.cs = splrep(self.model.wave, med_sub(self.model.flux))
        self.interpolation = interpolation
        if self.model is not None:
            self.inter = interp1d(self.model.wave, med_sub(self.model.flux), bounds_error=False,
                                  kind=self.interpolation, fill_value=0.0)
            # self.inter = InterpolatedUnivariateSpline(self.model.wave, med_sub(self.model.flux), k=3)
        self.set_norm = 'ones'
        self.sigma2 = 1. # CCF data normalization (defaults: 1., other: np.var(data, axis=0))
     
    
    @timeit
    def xcorr(self, f):
    #   return np.dot(med_sub(f) / np.var(f, axis=0), self.g)
        if self.set_norm == 'var':
            self.sigma2 = np.var(f, axis=0)
        return np.dot(f /self.sigma2, self.g)
    
    
    def __call__(self, datax, datay):
        # Ignore nans in data
        self.nans = np.isnan(datax)
        # Interpolate template to all RV shifts
        self.g = self.inter(np.outer(datax[~self.nans],self.beta))
        # Call function to calculate CCF-map
        jit_ccf = jit(self.xcorr)
        self.map = jit_ccf(datay[:,~self.nans])
        return self
    
    def imshow(self, ax=None, fig=None, title='', **kwargs):
        # plot y-axis as phase (if available)
        
        y1, y2 = 0, self.map.shape[0]-1
        if hasattr(self, 'phase'):
            y1, y2 = np.min(self.phase), np.max(self.phase)

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
    
    def save(self, outname):
        numpy.save(outname, self.__dict__)
        print('{:} saved...'.format(outname))
        return None
    def load(self, filename):
        print('Loading Datacube from...', filename)
        d = np.load(filename, allow_pickle=True).tolist()
        for key in d.keys():
            setattr(self, key, d[key])
        return self
    
class KpV(CCF):
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
    
    
    def get_planet_grid(self):
        test_planet = self.planet.copy()
        grid = []
        for ikp in range(len(self.kpVec)):
            test_planet.Kp = self.kpVec[ikp]
            grid.append(test_planet.RV)
        return np.asarray(grid)[:,:,np.newaxis]+self.vrestVec # RV_grid[iKp, iObs, iRV]
    
    def interpolate_ccf(self, iKp, iObs):
        return np.interp(self.RV_grid[iKp,iObs,:], self.ccf.RV, self.ccf.map[iObs,])

    @timeit
    def get_map(self):
        self.RV_grid = self.get_planet_grid() # RV_grid[iKp, iObs, iRV]
        
        ecl = False * np.ones_like(self.planet.RV)
        if hasattr(self.planet, 'eclipse_mask'):
            ecl = self.planet.eclipse_mask
            
        # Define the function to be jit-compiled
        jit_vfun = jit(vmap(self.interpolate_ccf, in_axes=(None,0), out_axes=0)) # for each Kp, interpolate the CCF
        ikp = np.arange(self.RV_grid.shape[0], dtype=int)
        # iObs = np.arange(self.RV_grid.shape[1], dtype=int)
        iObs = np.where(ecl==False)[0] # only use the non-eclipsed observations
        # Call function with jit-compiled vmap
        self.ccf_map = np.sum(jit_vfun(ikp, iObs), axis=0)
        return self
        
        
        
    @timeit
    def run(self, ignore_eclipse=True, ax=None):
        '''Generate a Kp-Vsys map
        if snr = True, the returned values are SNR (background sub and normalised)
        else = map values'''

        ecl = False * np.ones_like(self.planet.RV)
        if ignore_eclipse:
            ecl = self.planet.mask_eclipse(return_mask=True)

        # self.ccf_map = np.zeros((len(self.kpVec), len(self.vrestVec)))
        ccf_map = numpy.zeros((len(self.kpVec), len(self.vrestVec)))

        # interpolation function to vectorise with `vmap`
        fun = lambda x: np.interp(self.planet.RV[x]+self.vrestVec,
                                  self.ccf.RV, self.ccf.map[x,])
        vfun = vmap(fun)
        jit_vfun = jit(vfun)
        emask = np.where(ecl==False)[0]
        for ikp in range(len(self.kpVec)):
            self.planet.Kp = self.kpVec[ikp]
            # old way of doing it (slow)
            # for iObs in np.where(ecl==False)[0]:
                # ccf_map[ikp,] += interp1d(self.ccf.RV, self.ccf.map[iObs,])(outRV)
            ## New way (with JAX) ##
            # ccf_map[ikp,] = np.sum(jit(vfun)(emask), axis=0) # WARNING: DOES NOT WORK WITH JIT
            ccf_map[ikp,] = np.sum(vfun(emask), axis=0) # this works
            # self.ccf_map = self.ccf_map.at[ikp,].set(np.sum(vfun(emask), axis=0)) # this is slower than line above
        self.ccf_map = ccf_map
        
            

        self.bestSNR = self.snr.max() # store info as variable
        if ax != None: self.imshow(ax=ax)
        return self
    
    def plot(self, figsize=(6,6), peak=None, vmin=None, vmax=None,
                     outname=None, title=None, display=True, **kwargs):
        '''Plot Kp-Vsys map with horizontal and vertical slices 
        snr_max=True prints the SNR for the maximum value'''
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(6,6)
        gs.update(wspace=0.00, hspace=0.0)
        ax1 = fig.add_subplot(gs[1:5,:5])
        ax2 = fig.add_subplot(gs[:1,:5])
        ax3 = fig.add_subplot(gs[1:5,5])
        # ax2 = fig.add_subplot(gs[0,1])
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        ax3.xaxis.tick_top()
        
        eps = 0.1 * (self.snr.max()-self.snr.max())
        vmin = vmin or self.snr.min() - eps
        vmax = vmax or self.snr.max() + eps
        
        ax2.set_ylim(vmin, vmax)
        ax3.set_xlim(vmin, vmax)
            
        lims = [self.vrestVec[0],self.vrestVec[-1],self.kpVec[0],self.kpVec[-1]]

        obj = ax1.imshow(self.snr,origin='lower',extent=lims,aspect='auto', 
                         cmap='inferno', vmin=vmin, vmax=vmax)
    
        # figure settings
        ax1.set(ylabel='$K_p$ (km/s)', xlabel='$\Delta v$ (km/s)', **kwargs)
        
        # colorbar
        cax = fig.add_axes([ax3.get_position().x1+0.01,ax3.get_position().y0,
                            0.035,ax3.get_position().height])

        fig.colorbar(obj, cax=cax)
        
        if peak is None:
            peak = self.snr_max()
       # get the values     
        self.snr_at_peak(peak)
    
        row = self.kpVec[self.indh]
        col = self.vrestVec[self.indv]
        print('Horizontal slice at Kp = {:.1f} km/s'.format(row))
        print('Vertical slice at Vrest = {:.1f} km/s'.format(col))
        ax2.plot(self.vrestVec, self.snr[self.indh,:], 'gray')
        ax3.plot(self.snr[:,self.indv], self.kpVec,'gray')
        
        
    
        line_args = {'ls':':', 'c':'white','alpha':0.35,'lw':'3.', 'dashes':(0.7, 1.)}
        ax1.axhline(y=row, **line_args)
        ax1.axvline(x=col, **line_args)
        ax1.scatter(col, row, marker='*', c='red',label='SNR = {:.2f}'.format(self.peak_snr), s=6.)
        ax1.legend(handlelength=0.75)

    
        if title != None:
            fig.suptitle(title, x=0.45, y=0.915, fontsize=14)
    
        if outname != None:
            fig.savefig(outname, dpi=200, bbox_inches='tight', facecolor='white')
        if not display:
            plt.close()
        return self
    
    def snr_max(self, display=False):
        # Locate the peak
        self.bestSNR = self.snr.max()
        ipeak = np.where(self.snr == self.bestSNR)
        bestVr = float(self.vrestVec[ipeak[1]])
        bestKp = float(self.kpVec[ipeak[0]])
        
        if display:
            print('Peak position in Vrest = {:3.1f} km/s'.format(bestVr))
            print('Peak position in Kp = {:6.1f} km/s'.format(bestKp))
            print('Max SNR = {:3.1f}'.format(self.bestSNR))
        return(bestVr, bestKp, self.bestSNR)
    
    def snr_at_peak(self, peak=None):
        '''
        FInd the position and the SNR value of a given peak. If `peak` is a float
        it is considered the Kp value and the function searches for the peak around a range of DeltaV (< 5km/s)
        If `peak` is None, then we search for the peak around the expected planet position with a range of
        +- 10 km/s for Kp
        +- 5 km/s for DeltaV

        Parameters
        ----------
        peak : None, float, tuple, optional
            Position of the peak. The default is None.

        Returns
        -------
            self (with relevant values stored as self.peak_pos and self.peak_snr)

        '''
        if peak is None:
            snr = self.snr
            mask_kp = np.abs(self.kpVec - self.kpVec.mean()) < 10.
            mask_dv = np.abs(self.vrestVec) < 5. # around 0.0 km/s
            snr[~mask_kp, :] = snr.min()
            snr[:, ~mask_dv] = snr.min()

            # max_snr = self.snr[mask_kp, mask_dv].argmax()
            indh,indv = np.where(snr == snr.max())
            self.indh, self.indv = int(indh), int(indv)
            self.peak_pos = (float(self.vrestVec[self.indv]), float(self.kpVec[self.indh]))

        elif isinstance(peak, float):
            self.indh = np.abs(self.kpVec - peak).argmin()
            mask_dv = np.abs(self.vrestVec) < 5. # around 0.0 km/s
            mask_indv = self.snr[self.indh, mask_dv].argmax()
            self.indv = np.argwhere(self.vrestVec == self.vrestVec[mask_dv][mask_indv])
            print(self.vrestVec[self.indv])

        elif isinstance(peak, (tuple, list)):
            self.indv = np.abs(self.vrestVec - peak[0]).argmin()
            self.indh = np.abs(self.kpVec - peak[1]).argmin()

        self.peak_snr = float(self.snr[self.indh,self.indv])
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
    
    