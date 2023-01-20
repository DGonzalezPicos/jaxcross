import jax.numpy as np
import numpy
from astropy.io import fits
import matplotlib.pyplot as plt

class CRIRES:
    def __init__(self, files):
        self.files = files
        
    # def __str__(self):
    #     return f"CRIRES"
    # def __repr__(self):
    #     return f"CRIRES object with {len(self.files)} files."
    
        
    def read(self):
        print(f'Reading files ({len(self.files)})...')
        wave_list, flux_list, flux_err_list = ([] for _ in range(3))
        for i,file in enumerate(self.files):
            w,f,err = self.__load_fits(file)
            wave_list.append(w), flux_list.append(f), flux_err_list.append(err)
        self.wave = np.array(wave_list)
        self.flux = np.array(flux_list)
        self.flux_err = np.array(flux_err_list)
            
        return self
    
    @staticmethod
    def __load_fits(fits_file):
        """Read FITS file containing a single exposure
        Return wavelength, flux, and flux error arrays with dimension (detector, order, pixel)

        Args:
            fits_file (path): Path to file.

        Returns:
            wave: array
            flux: array
            flux_err: array
        """
        with fits.open(fits_file) as hdul:
            # hdul.info()
            wave, flux, flux_err = (numpy.zeros((3, 7, 2048)) for _ in range(3)) # (detector, order, pixel)
            for nDet in range(3):
                columns = hdul[nDet+1].columns
                data = hdul[nDet+1].data
                wave[nDet,] = numpy.array([data.field(key) for key in columns.names if key.endswith("WL")])
                flux[nDet,] = numpy.array([data.field(key) for key in columns.names if key.endswith("SPEC")])
                flux_err[nDet,] = numpy.array([data.field(key) for key in columns.names if key.endswith("ERR")])
        swap = lambda x: numpy.swapaxes(x, 0, 1) # swap detector and order axes
        return swap(wave), swap(flux), swap(flux_err)
    
    def imshow(self, nOrder, nDet, ax=None, fig=None, **kwargs):
        wave, flux = self.wave[:,nOrder, nDet,:], self.flux[:,nOrder, nDet,:]
        ax = ax or plt.gca()
        y1, y2 = 0, flux.shape[0]
        ext = [np.nanmin(wave), np.nanmax(wave), y1, y2]
        im = ax.imshow(flux,origin='lower',aspect='auto',
                        extent=ext, **kwargs)
        if not fig is None: fig.colorbar(im, ax=ax, pad=0.05)
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='white')
        return im
    def plot_master(self, nOrder, nDet, ax=None, **kwargs):
        wave, flux = self.wave[:,nOrder, nDet,:], self.flux[:,nOrder, nDet,:]
        ax = ax or plt.gca()
        ax.plot(np.median(wave, axis=0), np.median(flux, axis=0), **kwargs)
        return None
    
    def trim(self, x1=0, x2=0):
        
        fun1 = lambda x: x.at[:,:,:,:x1].set(np.nan)
        fun2 = lambda x: x.at[:,:,:,-x2:].set(np.nan)
            
        for attr in ['wave', 'flux', 'flux_err']:
            setattr(self, attr, fun2(fun1(self.__dict__[attr])))

        return self
    
    def normalise(self):
        med = np.nanmedian(self.flux, axis=0, keepdims=True)
        self.flux = self.flux / med
        self.flux_err = self.flux_err / med
        return self
    
    
    def PCA(self, N=1, nOrder=0, nDet=0, ax=None):
        '''PCA decomposition on reconstruction with the first N components removed
        Implemented in `numpy` for now. JAX `jnp.linalg.svd` is not working...'''
        f = self.flux[:,nOrder, nDet,:]
        self.nans = numpy.isnan(f[0,])
        # sub_med = lambda x: numpy.subtract(x, np.nanmedian(x))
        f_nonans = f[:,~self.nans]
        f_nonans = (f_nonans.T - numpy.nanmedian(f_nonans, axis=1)).T
        print(f_nonans.shape)

        # Compute full SVD
        u, s, vh = numpy.linalg.svd(f_nonans, full_matrices=False, compute_uv=True)
        s1 = numpy.copy(s)
        s1[0:N] = 0.
        W=numpy.diag(s1)
        f_rec = numpy.dot(u,numpy.dot(W,vh))
        
        new_f = f.at[:,~self.nans].set(f_rec)
        self.flux = self.flux.at[:,nOrder, nDet,:].set(new_f)
        
        if ax is not None: self.imshow(ax=ax, nOrder=nOrder, nDet=nDet)
        return self
    
    def gaussian_filter(self, window=15., nOrder=0, nDet=0, ax=None):
        from scipy import ndimage
        
        f = self.flux[:,nOrder, nDet,:]
        f_nonans = f[:,~self.nans]
        
        lowpass = ndimage.gaussian_filter(f_nonans, [0, window])
        
        new_f = f.at[:,~self.nans].set(f_nonans-lowpass)
        self.flux = self.flux.at[:,nOrder, nDet,:].set(new_f)
        if ax is not None: self.imshow(ax=ax, nOrder=nOrder, nDet=nDet)
        return self
        

if __name__ == '__main__':
    
    import pathlib
    path = pathlib.Path("/home/dario/phd/pycrires/pycrires/product/obs_staring/")
    files = sorted(path.glob("cr2res_obs_staring_extracted_*.fits"))
    
    nOrder, nDet = 1,1
    crires = CRIRES(files).read()

    fig, ax = plt.subplots(5,1,figsize=(10,4))
    crires.imshow(nOrder, nDet, ax=ax[0])
    
    crires.trim(20,20)
    crires.imshow(nOrder, nDet, ax=ax[1])
    crires.normalise()
    crires.imshow(nOrder, nDet, ax=ax[2])
    crires.PCA(4, 0, 0, ax=ax[3])
    crires.gaussian_filter(15, 0, 0, ax=ax[4])
    plt.show()

