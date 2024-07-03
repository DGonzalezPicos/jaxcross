# import jax.numpy as np
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import copy
from .eso_utils import Header
from astropy.stats import sigma_clip

eso = Header(tel_unit=3) # CRIRES+ is mounted on VLT UT3 (Melipal)

class CRIRES:
    def __init__(self, files=None, target='target'):
        '''CRIRES data object
        
        Parameters
        ----------------
            files = list of FITS files to CRIRES reduced data 
            target = name of target (default: 'target')
        '''
        self.files = files
        self.target = target
        
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
    def nFrames(self):
        '''Alias for nObs'''
        return self.nObs
    
    @property
    def nOrder(self):
        if hasattr(self, 'iOrder'):
            return 1
        else:
            return self.flux.shape[1]
    @property
    def nOrders(self):
        '''Alias for nOrder'''
        return self.nOrder
    @property
    def nDet(self):
        if hasattr(self, 'iDet'):
            return 1
        return self.flux.shape[2]
    
    @property
    def nPix(self):
        return self.wave.shape[-1]
    
    @property
    def snr(self):
        '''According to the CRIRES+ manual:
        *the spectrum can be directly divided by the error-spectrum 
        to obtain the signal-to-noise ratio*'''
        snr = np.zeros_like(self.flux)
        shape = self.flux.shape
        
        self.flux_err[self.flux_err == 0] = np.nanpercentile(self.flux_err[self.flux_err>0], 95) # avoid divide by zero
        err_nans = np.isnan(self.flux_err)
        flux_nans = np.isnan(self.flux)
        
        nans = np.logical_or(err_nans, flux_nans)
        err = self.flux_err[~nans]
    
        f = self.flux[~nans]
        snr[~nans] = np.abs(np.divide(f, err))
        
        return snr
    
    @property
    def master(self):
        '''Weighted time-average of all spectra'''
        # snr2 = np.nanmean(self.self.flux_err, axis=3)**2 # weights per (frame, order, detector)
        sigma2 = np.nanmean(np.power(self.flux_err, 2), axis=-1) # weights per (frame, order, detector)
        sigma2[sigma2 == 0] = np.nanpercentile(sigma2[sigma2>0], 1) # avoid divide by zero

        weights = sigma2 / np.sum(sigma2)
        self.weights = np.expand_dims(weights, axis=-1)
        
        return np.sum(self.flux*self.weights, axis=0)
    
    @property
    def master_err(self):
        '''Error propagation for master spectrum'''
        if not hasattr(self, 'master'): 
            _ = self.master
        # assert hasattr(self, 'weights'), 'Run `master` first'
        return np.sqrt(np.sum(np.power(self.flux_err, 2)*self.weights**2, axis=0))
    @property
    def master_snr(self):
        self.master_err[self.master_err == 0] = np.nanpercentile(self.master_err[self.master_err>0], 5) # avoid divide by zero
        return self.master / self.master_err
    
    
    
    @property
    def bad_pixels(self):
        return np.isnan(self.flux)

        
        
    
        
        
    def read(self):
        print(f'Reading files ({len(self.files)})...')
        wave_list, flux_list, flux_err_list = ([] for _ in range(3))
        # ext_header = {k:[] for k in eso.keys()} # Create empty dict to extract header info 

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
    
    def load_fits2D(self, fits_file):
        swap = lambda x: np.swapaxes(x, 0, 1) # swap detector and order axes

        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            data = map(swap, (hdul[i].data for i in range(1, len(hdul))))
            
            keys = iter(['flux', 'flux_err', 'wave', 'wave_corr'])
            for val in data:
                key = next(keys)
                print('Setting attribute: ', key, '...')
                setattr(self, key, val)
            # shape = (order, detector, slit_frac, pixel)
            
        return self
    
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
        
    def plot(self, frame=0, ax=None, lw=0.9, **kwargs):
        '''Plot a single spectrum given (order, det)'''
        assert hasattr(self, 'iOrder'), 'Order not set'
        assert hasattr(self, 'iDet'), 'Detector not set'
        ax = ax or plt.gca()
        ax_snr = ax.twinx()
        
        if isinstance(frame, str):
            frame = np.arange(self.nFrames)
        frames = [frame] if isinstance(frame, int) else frame
        cmap = plt.cm.get_cmap('viridis', len(frames))
        colors = iter(cmap(np.linspace(0, 1, len(frames))))
        
        for f in frames:
            ax.plot(self.wave[f,:], self.flux[f,:], 
                    label=f'Frame {f}', c=next(colors), **kwargs)
            ax_snr.plot(self.wave[f,:], self.snr[f,:], alpha=0.0)
        
        ax_snr.set_ylabel('SNR')
        ax.set(xlabel='Wavelength (nm)', ylabel='Flux [ADU]', title=f'Order {self.iOrder} -- Det {self.iDet}')
        ax.legend()
        return None
        
        
    def plot_all_range(self, frame=0, trim=10, outname=None):
        assert trim > 0, 'trim must be > 0'
        assert frame < self.nFrames, f'frame must be < {self.nFrames}'

        fig, ax = plt.subplots(1, figsize=(9, 4))
        cmap = plt.get_cmap('inferno', self.nDet*self.nOrders)
        colors = iter(cmap(np.linspace(0, 1, self.nDet*self.nOrders)))

        
        for iOrder in range(self.nOrders):
            for iDet in range(self.nDet):
                ax.plot(self.wave[frame,iOrder,iDet,trim:-trim], self.flux[frame,iOrder,iDet, trim:-trim], c=next(colors), lw=0.5)

        ax.set(ylabel='Counts', xlabel='Wavelength (nm)', title=f'CRIRES+ {self.target}\n frame = {frame}')
        plt.show()
        if outname is not None:
            fig.savefig(outname, dpi=300, bbox_inches='tight')
        return None
            
    def plot_all_orders(self, frame=0, det=0, trim=10, outname=None):
        '''Plot all orders from a given detector and frame'''
        assert trim > 0, 'trim must be > 0'
        if isinstance(frame, int):
            assert frame < self.nFrames, f'frame must be < {self.nFrames}'
        assert det < self.nDet, f'detector must be < {self.nDet}'
        
        cmap = plt.get_cmap('inferno')
        colors = cmap(np.linspace(0, 1, self.nOrder*2))
        fig, ax = plt.subplots(self.nOrder, figsize=(12, 14))
        for iOrder in range(self.nOrder):
            data = self.order(iOrder).detector(det)
            if isinstance(frame, int):
                select = lambda x: x[frame, trim:-trim]
                x, y, yerr, snr = map(select, [data.wave, data.flux, data.flux_err, data.snr])
            elif frame in ['combined', 'master']:
                select = lambda x: np.nanmedian(x[:, trim:-trim], axis=0)
                x = select(data.wave)
                y = data.master[trim:-trim]
                yerr = data.master_err[trim:-trim]
                snr = y / yerr
                
                
            
            ax[iOrder].plot(x,y, label='Order {}'.format(iOrder),color=colors[iOrder])
            ax[iOrder].fill_between(x, y-yerr, y+yerr, alpha=0.4, color=colors[iOrder])
            tlabel = 'Telluric Model' if iOrder == 0 else ''
            self.plot_telluric(x1=x.min(), x2=x.max(), ax=ax[iOrder], scale=np.nanmean(y), 
                               color='limegreen', alpha=0.8, ls='--', label=tlabel)
                
            ax_snr = ax[iOrder].twinx()
            ax_snr.plot(x, snr, color=colors[iOrder], alpha=0.0)
            ax_snr.set(ylabel='SNR')  
               
            ax[iOrder].legend()
            x1, x2 = x.min(), x.max()
            ax[iOrder].set_xlim(x1, x2)
            if iOrder == 3:
                ax[iOrder].set(ylabel='Flux [ADU]') 
            
        ax[0].set_title(f'CRIRES+ {self.target}\n Detector {det} -- Frame {frame}', fontsize=14)
        ax[len(ax)-1].set_xlabel('Wavelength [nm]')
        plt.show()
        if outname is not None:
            fig.savefig(outname, dpi=300, bbox_inches='tight')
        return None
    
    def plot_telluric(self, x1=None, x2=None, ax=None, scale=1., **kwargs):
        ax = ax or plt.gca()
        file = "/home/dario/phd/jaxcross/data/transm_spec.dat" # my file

        x1 = x1 if x1 is not None else self.wave.min()
        x2 = x2 if x2 is not None else self.wave.max()
        
        twave, trans = np.loadtxt(file, unpack=True)
        mask = (twave > x1) & (twave < x2)
        ax.plot(twave[mask], trans[mask] * scale, **kwargs)
        return None
    
    
    def plot_master(self, ax=None, snr=False, **kwargs):
        ax = ax or plt.gca()
        
        x = self.wavesol[~self.nans] if hasattr(self, 'wavesol') else np.nanmedian(self.wave, axis=0)[~self.nans]
        y = self.master[~self.nans] if not snr else self.master_snr[~self.nans]
        ax.plot(x, y, **kwargs)
        return None
    
    def plot_spectra(self, ax=None, offset=0.0, snr=False, **kwargs):
        ax = ax or plt.gca()
        n = self.flux.shape[0]
        cmap = plt.cm.get_cmap('viridis', n)
        colors = iter([cmap(x) for x in range(n)])
        if not hasattr(self, 'wavesol'):
            self.set_wavesol()
            
        for i,f in enumerate(self.flux):
            y = f[~self.nans] if not snr else self.snr[i, ~self.nans]
            ax.plot(self.wavesol[~self.nans], offset + y, c=next(colors), **kwargs)
        return self
    
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
    
    def align(self, ref_frame=0, cycles=3, ax=None):
        '''Align the FRAMES to a reference FRAME using cross-correlation.'''
        from align import Align
        self_copy = self.copy()
        al = Align(self_copy.wave, self_copy.flux, self_copy.flux_err).align_all(ref_frame, cycles)
        self.wave = al.wave
        self.flux = al.flux
        self.flux_err = al.flux_err
        return self
    
    def align_all_pairs(self, ref_frame=0, cycles=3, ax=None):
        '''Align the FRAMES to a reference FRAME using cross-correlation.'''
        from align import Align
        for iOrder in range(self.nOrders):
            for iDet in range(self.nDet):
                pair = self.order(iOrder).detector(iDet)
                pair_aligned = Align(pair.wave, pair.flux, pair.flux_err).align_all(ref_frame, cycles)
                
                self.update_pair(pair_aligned, iOrder=iOrder, iDet=iDet)
            
        return self
    
    def update_pair(self, self_pair, iOrder=None, iDet=None):
        '''Update the current object with the values of the input 
        single-order, single-detector object.'''
        keys = ['wave', 'flux', 'flux_err']
        # assert hasattr(self, 'iOrder') and hasattr(self, 'iDet')
        # iOrder, iDet = self_pair.iOrder, self_pair.iDet
        for k in keys:
            getattr(self, k)[iOrder,iDet] = getattr(self, k)
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
    
    def order(self, iOrder, axis=1):
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
    def trim(self, x1=0, x2=None, ax=None):
        '''Trim edges of spectrum
        
        Parameters
        ----------
            x1 : int, optional
            x2 : int, optional
            ax : matplotlib axis, optional
        
        Return
        ------
            Trimmed CRIRES object (self)'''
        
        x2 = x2 or x1 # by default x2=x1 
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
    
    def save_txt(self, outname):
        assert len(self.wavesol.shape) == 3, "Wavelength solution is not 3D (order, det, pix)"
        assert len(self.master.shape) == 3, "Flux is not 3D (order, det, pix)"
        
        wave = self.wavesol.flatten()
        flux = self.master.flatten()
        err = self.master_err.flatten()
        orders = np.array([i * np.ones(self.nPix) for i in range(self.nOrders)], dtype=int).flatten()
        orders = np.tile(orders, self.nDet)
        dets = np.array([i * np.ones(self.nPix) for i in range(self.nDet)], dtype=int).flatten()
        dets = np.tile(dets, self.nOrders)
        print(dets.shape)
        print(dets)
        np.savetxt(outname, np.array([wave, flux, err, orders, dets]).T, 
                   header='Wavelength (nm), Flux, Error, Order, Detector',
                   fmt=['%.8f', '%.8f', '%.8f', '%i', '%i'])
        print('{:} saved...'.format(outname))
        return None
    
    
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
    
    # data = self.order(iOrder).detector(iDet) 
    # print(data.flux.shape) # (100, 2048)

    # fig, ax = plt.subplots(5,1,figsize=(10,4))
    # data.imshow(ax=ax[0])

    # data.trim(20,20, ax=ax[1])
    # data.normalise(ax=ax[2])
    # data.imshow(ax=ax[2])
    # data.PCA(4, ax=ax[3])
    # data.gaussian_filter(15, ax=ax[4])
    # plt.show()
