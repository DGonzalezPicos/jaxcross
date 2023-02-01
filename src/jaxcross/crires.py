import jax.numpy as np
import numpy
from astropy.io import fits
import matplotlib.pyplot as plt
import copy

class CRIRES:
    def __init__(self, files=None):
        self.files = files
        
    # def __str__(self):
    #     return f"CRIRES"
    # def __repr__(self):
    #     return f"CRIRES object with {len(self.files)} files."
    
    @property
    def nPix(self):
        return self.wave.shape[-1]
    @property
    def nObs(self):
        return self.flux.shape[0]
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
        return swap(wave), swap(flux), swap(flux_err)#, header
    
    def copy(self):
        return copy.deepcopy(self)
    
    def imshow(self, ax=None, fig=None, **kwargs):
        ax = ax or plt.gca()
        y1, y2 = 0, self.flux.shape[0]
        ext = [np.nanmin(self.wave), np.nanmax(self.wave), y1, y2]
        im = ax.imshow(self.flux,origin='lower',aspect='auto',
                        extent=ext, **kwargs)
        if not fig is None: fig.colorbar(im, ax=ax, pad=0.05)
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='white')
        return im
    def plot_master(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        ax.plot(np.median(self.wave, axis=0), np.median(self.flux, axis=0), **kwargs)
        return None
    
    def order(self, iOrder):
        N_orders = self.flux.shape[1]
        assert iOrder < N_orders, f"Order {iOrder} does not exist. Max order is {N_orders-1}"
        self_copy = self.copy()
        select = lambda x: x[:,iOrder,:,:]
        for attr in ['wave', 'flux', 'flux_err']:
            setattr(self_copy, attr, select(self.__dict__[attr]))
        self_copy.iOrder = iOrder
        return self_copy
    
    def detector(self, iDet):
        assert hasattr(self, 'iOrder'), "Select order first >> self.order(iOrder)"
        Ndet = self.flux.shape[1]
        assert iDet < Ndet, f"Detector must be an integer between 0 and {Ndet-1}"
        self_copy = self.copy()
        select = lambda x: x[:,iDet,:]
        for attr in ['wave', 'flux', 'flux_err']:
            setattr(self_copy, attr, select(self.__dict__[attr]))
        self_copy.iDet = iDet
        return self_copy
    
    def check_wavesol(self, ax=None):
        print('Checking wavelength solution...')
        ax = ax or plt.gca()
        wavesol = np.median(self.wave, axis=0)
        wavesol_std = np.std(self.wave, axis=0)
        ax.plot(wavesol, wavesol_std, '--o', ms=0.5)
        print('Wave Solution Error: Min {:.2e} / Mean {:.2e} / Max {:.2e} nm'.format(
            np.nanmin(wavesol_std), np.nanmedian(wavesol_std), np.nanmax(wavesol_std)))
        return None
    
    def clip(self, sigma=5.):
        nans = np.isnan(self.flux[0,])
        mask = np.abs(self.flux[:,~nans] - np.nanmedian(self.flux, axis=0)) > sigma * np.nanstd(self.flux, axis=0)
        
        self.flux[mask] = np.nan
    def trim(self, x1=0, x2=0, ax=None):
        
        fun1 = lambda x: x.at[:,:x1].set(np.nan)
        fun2 = lambda x: x.at[:,-x2:].set(np.nan)
            
        for attr in ['wave', 'flux', 'flux_err']:
            setattr(self, attr, fun2(fun1(self.__dict__[attr])))
        self.__check_ax(ax)
        return self
    
    def normalise(self, ax=None):
        med = np.nanmedian(self.flux, axis=1)
        self.flux = (self.flux.T / med).T
        self.flux_err = (self.flux_err.T / med).T
        self.__check_ax(ax)
        return self
    
    def __check_ax(self, ax):
        if ax != None:
            self.imshow(ax=ax)
        return None    
    
    def __check_dims(self):
        assert hasattr(self, 'iOrder'), "Select order first >> self.order(iOrder)"
        assert hasattr(self, 'iDet'), "Select detector first >> self.detector(iDet)"
        return None
    
    def PCA(self, N=1, nOrder=0, nDet=0, ax=None):
        '''PCA decomposition on reconstruction with the first N components removed
        Implemented in `numpy` for now. JAX `jnp.linalg.svd` is not working...'''
        self.__check_dims()
        self.nans = numpy.isnan(self.flux[0,])
        # sub_med = lambda x: numpy.subtract(x, np.nanmedian(x))
        f_nonans = self.flux[:,~self.nans]
        f_nonans = (f_nonans.T - numpy.nanmedian(f_nonans, axis=1)).T
        print(f_nonans.shape)

        # Compute full SVD
        u, s, vh = numpy.linalg.svd(f_nonans, full_matrices=False, compute_uv=True)
        s1 = numpy.copy(s)
        s1[0:N] = 0.
        W=numpy.diag(s1)
        f_rec = numpy.dot(u,numpy.dot(W,vh))
        
        self.flux = self.flux.at[:,~self.nans].set(f_rec)
        # self.flux = self.flux.at[:,nOrder, nDet,:].set(new_f)
        
        if ax is not None: self.imshow(ax=ax)
        return self
    
    def gaussian_filter(self, window=15., ax=None):
        from scipy import ndimage
        self.__check_dims()
        f_nonans = self.flux[:,~self.nans]
        
        lowpass = ndimage.gaussian_filter(f_nonans, [0, window])
        
        self.flux = self.flux.at[:,~self.nans].set(f_nonans-lowpass)
        # self.flux = self.flux.at[:,nOrder, nDet,:].set(new_f)
        if ax is not None: self.imshow(ax=ax)
        return self
    
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
    
    def inject_signal(self, planet, template, RV=None, factor=1., ax=None):
        temp = template.copy()
        p = planet.copy()
        p.frame = self.frame

        if factor > 1.: 
            temp = temp.copy().boost(factor)

        # get 2D template shifted at given RV or at planet.RV if no RV vector is passed
        if RV is None:
            RVt = p.RV
        else:
            RVt = RV*np.ones_like(p.RV)

        temp = temp.shift_2D(RVt, self.wave)

        # inject only for out-of-eclipse frames
        emask = p.mask_eclipse(return_mask=True)
        data_masked = self.flux[~emask,:]
        # data_err_masked = self.flux_err[~mask,:]
        temp_masked = temp.gflux[~emask,:]
        self.flux = self.flux.at[~emask,:].set(data_masked * temp_masked)
        

        if ax != None: self.imshow(ax=ax)
        return self
    
        

if __name__ == '__main__':
    import pathlib
    path = pathlib.Path("/home/dario/phd/pycrires/pycrires/product/obs_staring/")
    files = sorted(path.glob("cr2res_obs_staring_extracted_*.fits"))
    
    with fits.open(files[0]) as hdul:
        hdul.info()
        header = hdul[0].header
    hdr_keys = ['RA','DEC','MJD-OBS']
    air_key = ['HIERARCH ESO TEL AIRM '+i for i in ['START','END']]
    airmass = numpy.round(numpy.mean([header[key] for key in air_key]), 3)
    
    my_header = {key:header[key] for key in hdr_keys}
    my_header['AIRMASS'] = airmass
    # iOrder, iDet = 1,1
    # crires = CRIRES(files).read()
    
    # data = crires.order(iOrder).detector(iDet) 
    # print(data.flux.shape) # (100, 2048)

    # fig, ax = plt.subplots(5,1,figsize=(10,4))
    # data.imshow(ax=ax[0])

    # data.trim(20,20, ax=ax[1])
    # data.normalise(ax=ax[2])
    # data.imshow(ax=ax[2])
    # data.PCA(4, ax=ax[3])
    # data.gaussian_filter(15, ax=ax[4])
    # plt.show()
