# import jax.numpy as np
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import copy
from .eso_utils import Header
from astropy.stats import sigma_clip

eso = Header(tel_unit=3) # CRIRES+ is mounted on VLT UT3 (Melipal)

class CRIRES:
    def __init__(self, files=None):
        self.files = files
        
        
        
        # plotting options for imshow()
        self.imshow_dict = {}
        self.imshow_dict['xstr'] = 0.75
        self.imshow_dict['ystr'] = 0.75
        
        # bookkeeping
        self._files_read = False
        
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        out = []
        out.append(f"{'-'*20} CRIRES data object {'-'*20}")
        out.append(f"{'Frames:':<10} {len(self.files):<10}")
        
        if self._files_read:
            out.append(f"{'Orders:':<10} {self.nOrder:<10}")
            out.append(f"{'Detectors:':<10} {self.nDet:<10}")
            out.append(f"{'Npix:':<10} {self.nPix:<10}")
        
        return '\n'.join(out)
    


    @property
    def nObs(self):
        return self.flux.shape[0]
    
    @property
    def nOrder(self):
        if hasattr(self, 'iOrder'):
            return 1
        else:
            return self.flux.shape[1]
    @property
    def nDet(self):
        if hasattr(self, 'iDet'):
            return 1
        return self.flux.shape[2]
    
    @property
    def nPix(self):
        return self.wave.shape[-1]
        
    
        
        
    def read(self):
        print(f'Reading files ({len(self.files)})...')
        wave_list, flux_list, flux_err_list = ([] for _ in range(3))
        ext_header = {k:[] for k in eso.keys()} # Create empty dict to extract header info 

        for i,file in enumerate(self.files):
            w,f,err = self.__load_fits(file)
            wave_list.append(w), flux_list.append(f), flux_err_list.append(err)
        self.wave = np.array(wave_list)
        self.flux = np.array(flux_list)
        self.flux_err = np.array(flux_err_list)
        self._files_read = True
        return self
    
    def set_header(self, ext_header):
        """Set header info from `ext_header` dict."""
        # Store the header info arrays in `self`
        for keys, vals in ext_header.items():
            if keys in eso.unique_keys:
                setattr(self, keys, np.unique(vals)[0])
            else:
                setattr(self, keys, np.array(vals))
        return self
    
    def __load_fits(self, fits_file):
        """Read FITS file containing a single exposure
        Return wavelength, flux, and flux error arrays with dimension (detector, order, pixel)

        Args:
            fits_file (path): Path to file.

        Returns:
            wave: array
            flux: array
            flux_err: array
        """
        ext_header = {k:[] for k in eso.keys()} # Create empty dict to extract header info
        with fits.open(fits_file) as hdul:
            # hdul.info()
            header = hdul[0].header
            wave, flux, flux_err = (np.zeros((3, 7, 2048)) for _ in range(3)) # (detector, order, pixel)
            # Extract header info
            # for k in ext_header.keys():
            #     ext_header[k].append(header[eso.dict[k]])
                
            for nDet in range(3):
                columns = hdul[nDet+1].columns
                data = hdul[nDet+1].data
                wave[nDet,] = np.array([data.field(key) for key in columns.names if key.endswith("WL")])
                flux[nDet,] = np.array([data.field(key) for key in columns.names if key.endswith("SPEC")])
                flux_err[nDet,] = np.array([data.field(key) for key in columns.names if key.endswith("ERR")])
        swap = lambda x: np.swapaxes(x, 0, 1) # swap detector and order axes
        # self.set_header(ext_header) # Store header info in `self`
        # TODO: implement header class for CRIRES in `eso_utils.py` with correct keys
        return swap(wave), swap(flux), swap(flux_err)#, header
    
    def copy(self):
        return copy.deepcopy(self)
    
    def imshow(self, ax=None, fig=None, label='', 
                xscale='linear', **kwargs):
            ax = ax or plt.gca()
            
            if not hasattr(self, 'extent'):
                y1, y2 = 0, self.flux.shape[0]
                x1, x2 = np.nanmin(self.wave), np.nanmax(self.wave)
                self.extent = [x1,x2, y1,y2]
                
            if hasattr(self, 'phase'):
                y1, y2 = self.cphase.min(), self.cphase.max()
                self.extent[2:] = [y1,y2]
                    
            # self.flux[:,self.nans] = np.nan
            im = ax.imshow(self.flux,origin='lower',aspect='auto',
                            extent=self.extent, **kwargs)
            ax.set_xscale(xscale)

            if len(label) > 0.:
                sigma = np.nanstd(self.flux)
                label_str = label +f'\n <$\sigma$> = {sigma:.3e}'
                ax.text(self.imshow_dict["xstr"], self.imshow_dict["ystr"], label_str, transform=ax.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.5))
                
            if not fig is None: fig.colorbar(im, ax=ax, pad=0.05)
            current_cmap = plt.cm.get_cmap()
            current_cmap.set_bad(color='white')
            return im
    
    def plot_master(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        if hasattr(self, 'wavesol'):
            wave = self.wavesol
        else:
            wave = np.nanmedian(self.wave, axis=0)
        ax.plot(wave[~self.nans], 
                np.nanmedian(self.flux, axis=0)[~self.nans], **kwargs)
        return None
    
    def sigma_clip(self, sigma=3, axis=0, replace_nans=True, ax=None):
        '''Sigma clip the flux array along the specified axis. Clipped values are set to NaN.
        NaN values are interpolated over using a Gaussian kernel.
        Automatically ignores the edges of the detector if present (i.e. does not interpolate these NaNs).'''
        flux = np.array(self.flux, dtype=np.float32) # convert to numpy
        if isinstance(axis, (list, tuple)):
            for ax_i  in axis:
                flux = sigma_clip(flux, sigma=sigma, axis=ax_i, masked=False, grow=0.)
        else:
            flux = sigma_clip(flux, sigma=sigma, axis=axis, masked=False, grow=0.)        
            
        self.flux = np.array(flux)
        if replace_nans:
            self.replace_nans()
        if ax is not None:
            self.imshow(ax=ax, label='Sigma clip {:.0f}σ'.format(sigma))
        return self
    
    def replace_nans(self, w=3., ax=None):
        from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
        # flux = np.array(self.flux, dtype=np.float64) # convert to numpy
        kernel = Gaussian2DKernel(w)
        if hasattr(self, 'ledge'):
            inter_flux = interpolate_replace_nans(self.flux[:,self.ledge:self.uedge], kernel)
            self.flux[:,self.ledge:self.uedge] = inter_flux
        else:
            self.flux = interpolate_replace_nans(self.flux, kernel)
            
        if ax is not None: self.imshow(ax=ax, label='Replace NaNs')
        return self
    
    def order(self, iOrder, axis=0):
        N_orders = self.flux.shape[1]
        assert iOrder < N_orders, f"Order {iOrder} does not exist. Max order is {N_orders-1}"
        self_copy = self.copy()
        select = lambda x: x.take(iOrder, axis=axis)
        for attr in ['wave', 'flux', 'flux_err']:
            setattr(self_copy, attr, select(self.__dict__[attr]))
        self_copy.iOrder = iOrder
        return self_copy
    
    def detector(self, iDet):
        assert hasattr(self, 'iOrder'), "Select order first >> self.order(iOrder)"
        Ndet = self.flux.shape[1]
        assert iDet < Ndet, f"Detector must be an integer between 0 and {Ndet-1}"
        self_copy = self.copy()
        # select = lambda x: x[:,iDet,:]
        select = lambda x: x.take(iDet, axis=1)
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
        '''Trim edges of spectrum
        
        Parameters
        ----------
            x1 : int, optional
            x2 : int, optional
            ax : matplotlib axis, optional
        
        Return
        ------
            Trimmed CRIRES object (self)'''
        
        self.wave[:,:x1] = np.nan
        self.wave[:,-x2:] = np.nan
        if hasattr(self, 'wavesol'):
            self.wavesol[:x1] = np.nan
            self.wavesol[-x2:] = np.nan
            
        if not ax is None: ax.imshow(ax, label='Trim ({:}:{:})'.format(x1, x2))
        return self
    
    def set_wavesol(self, wavesol=None):
        '''Set wavelength solution attribute `wavesol`
            New shape is (Npix,)
        
        Parameters
        ----------
            wavesol : array, optional
            
        Return 
        ----------
            CRIRES object with new attribute `wavesol`
        '''
        if wavesol is None:
            assert len(self.wave.shape) > 1, "Wavelength solution is not 2D"
            self.wavesol = np.median(self.wave, axis=0)
        else:
            self.wavesol = wavesol
        return self
    
    def normalise(self, ax=None):
        med = np.median(self.flux[:,~self.nans], axis=1)
        # divide by median
        self.flux[:, ~self.nans] /= med[:,None]
        self.flux_err[:,~self.nans] /= med[:,None]
        
        if ax is not None: self.imshow(ax, label='Normalised')
        return self
    
    def __check_ax(self, ax, label=''):
        if ax != None:
            self.imshow(ax=ax, label=label)
        return None    
    
    def __check_dims(self):
        assert hasattr(self, 'iOrder'), "Select order first >> self.order(iOrder)"
        # assert hasattr(self, 'iDet'), "Select detector first >> self.detector(iDet)"
        return None
    
    def PCA(self, n, mode='subtract', ax=None, save_model=False, standarize=False):
        '''Perform PCA on the data'''
        
        #TODO: properly implement STANDARIZATION (check out sklearn.preprocessing)
        # also useful tutorial in https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
        
        from sklearn import preprocessing
        # Step 0: Subtract mean of each channel
        f  = self.flux[:,~self.nans]
        f -= np.nanmedian(f, axis=1)[:,None]
        
        if standarize:
            # initialize StandardScaler
            sc = preprocessing.StandardScaler()

            # let it figure out means and standard deviations for each 
            # feature and use them to standardize the data
            print('Mean before scaling: ', f.mean(axis=0))
            print('Standard deviation before scaling: ', f.std(axis=0))
            sc.fit(f)
            f = sc.transform(f)
            print('Mean after scaling: ', f.mean(axis=0))
            print('Standard deviation after scaling: ', f.std(axis=0))
            
        # Step 1: Singular Value Decomposition (SVD)
        u, s, vh = np.linalg.svd(f, full_matrices=False)
        print(s.shape)
        s1, s2 = (np.copy(s) for _ in range(2))
        
        # data_pro, noise = (self.copy() for _ in range(2))
        data_pro, PCA_model = (self.flux.copy() for _ in range(2))
        
        # Step 2: Save the main PCs as the "processed data" in `data_pro`
        s1[0:n] = 0.
        W=np.diag(s1)
        data_pro[:,~self.nans] = np.dot(u,np.dot(W,vh))

        # we save the "discarded PCs" as `noise
        s2[n:] = 0
        W = np.diag(s2)
        PCA_model[:,~self.nans] = np.dot(u, np.dot(W, vh))
        
        if mode == 'subtract':  
            self.flux = 1.+ data_pro
        elif mode == 'divide':
            self.flux[:,~self.nans] /= (1. + PCA_model[:,~self.nans])
            
        if save_model:
            self.PCA_model = PCA_model
        
        
        if ax != None: self.imshow(ax=ax, label=f'PCA {n}')
        return self
    
    def gaussian_filter(self, window=15., mode='divide', ax=None):
        from scipy import ndimage
        lowpass = ndimage.gaussian_filter(self.flux[:,~self.nans], [0, window])
        
        self.flux[:,~self.nans] = getattr(np, mode)(self.flux[:,~self.nans], lowpass)
        if mode == 'divide':
            self.flux_err[:,~self.nans] /= lowpass
        
        if ax is not None: self.imshow(ax=ax, label=f'Gaussian Filter w={int(window)} pix')
        return self
    
    def save(self, outname):
        np.save(outname, self.__dict__)
        print('{:} saved...'.format(outname))
        return None
    def load(self, filename):
        print('Loading Datacube from...', filename)
        d = np.load(filename, allow_pickle=True).tolist()
        for key in d.keys():
            setattr(self, key, d[key])
        return self
    
    
    @property
    def nans(self):
        '''Returns a boolean array with **True** for NaNs along the wavelength axis'''
        wave = np.atleast_2d(self.wave) # View inputs as arrays with at least two dimensions.
        return np.isnan(wave).any(axis=0)
        
    
        

if __name__ == '__main__':
    import pathlib
    path = pathlib.Path("/home/dario/phd/pycrires/pycrires/product/obs_staring/")
    files = sorted(path.glob("cr2res_obs_staring_extracted_*.fits"))[:2]
    
    with fits.open(files[0]) as hdul:
        hdul.info()
        header = hdul[0].header
    hdr_keys = ['RA','DEC','MJD-OBS']
    air_key = ['HIERARCH ESO TEL AIRM '+i for i in ['START','END']]
    airmass = np.round(np.mean([header[key] for key in air_key]), 3)
    
    my_header = {key:header[key] for key in hdr_keys}
    my_header['AIRMASS'] = airmass
    # iOrder, iDet = 1,1
    crires = CRIRES(files).read()
    
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
