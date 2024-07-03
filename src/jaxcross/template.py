# import jax.numpy as np
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import copy

class Template:
    
    def __init__(self, wave=None, flux=None):
        self.wave = wave
        self.flux = flux
        
        self.wave_unit = 'nm' # TO-DO: implement auto-setting of units...
    # def read(self, fits_file, debug=True):
    #     with fits.open(fits_file) as hdul:
    #         if debug: hdul.info()
    #         self.wave = hdul[1].data
    #         self.flux = hdul[2].data
    #     return self

    
    def read(self, fits_file, debug=True):
        self.fits_file = fits_file
        with fits.open(fits_file) as hdul:
            if debug: hdul.info()
            self.header = hdul[0].header
            data = hdul[1].data
            print(data.shape)
            print(data.columns)
            self.wave = data['WAVE']
            self.flux = data['FLUX']
        return self
    
    def save(self, fits_file):
        hdu = fits.PrimaryHDU()
        hdu.header = self.header
        cols = [fits.Column(name='WAVE', format='D', array=self.wave),
                fits.Column(name='FLUX', format='D', array=self.flux)]
        tbhdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
        # tbhdu.header = self.header
        hdul = fits.HDUList([hdu, tbhdu])
        hdul.writeto(fits_file, overwrite=True)
        print(f'File saved to {fits_file}')
        return None
    
    
    def plot(self, ax=None, **kwargs):
        ax = ax or plt.gca()
        ax.plot(self.wave, self.flux, **kwargs)
        return None
    
    def sort(self):
        '''
        Sort `wlt` and `flux` vectors by wavelength.

        Returns
        sorted Template
        -------
        '''
        sort = np.argsort(self.wave)
        self.wave = self.wave[sort]
        self.flux = self.flux[sort]
        return self
    
    def crop(self, wmin, wmax):
        """ Crop the template to the given wavelength range.

        Args:
            wmin (float): Minimum wavelength. 
            wmax (float): Maxmimum wavelength.
        """
        crop = (self.wave >= wmin) & (self.wave <= wmax)
        self.wave = self.wave[crop]
        self.flux = self.flux[crop]
        return self
    def shift(self, RV):
        """ Shift the template by the given radial velocity.

        Args:
            RV (float): Radial velocity in km/s.
        """
        wave_s = self.wave * (1. - RV/299792.458)
        # self.flux = interp1d(self.wave, self.flux, kind='linear', bounds_error=False)(wave_s)
        self.wave = wave_s
        return self
    def copy(self):
        return copy.deepcopy(self)
    
    def boost(self, factor=10.):
        # mean = np.nanmean(self.flux)
        new_flux = self.flux - 1.
        new_flux *= factor
        self.flux = new_flux + 1.
        return self
    
    def shift_2D(self, RV, wave):
        """ Shift the template by the given radial velocity.

        Args:
            RV (float): Radial velocity in km/s.
        """
        beta = 1. - RV/299792.458
        wave_s = np.outer(beta, wave)
        self.gflux = interp1d(self.wave, self.flux, kind='linear', bounds_error=False,
                              fill_value=0.0)(wave_s)
        return self
    
    def convolve(self, fwhm=4.):
        from astropy.convolution import Gaussian1DKernel, convolve
        # FWHM = 2 * sqrt(2 * ln(2)) * sigma ~ 2.2 * sigma # Wikipedia FWHM
        g = Gaussian1DKernel(stddev=fwhm/2.355)
        self.flux = convolve(self.flux, g)
        return self
    
    def broad(self, resolution=100_000, vsini=0., in_place=False):
        ''' Broaden the template with a Gaussian instrumental profile and a 
        rotational broadening profile if vsini > 0.
        
        Parameters
        ------------
            resolution : float (default: 100_000)
                Instrumental resolution. R = dlambda/lambda
            vsini : float (default: 0.)
                Projected Rotational velocity in km/s.
                
        Returns
        ------------
            self : Template

        '''
        from PyAstronomy import pyasl
        self.resolution = resolution # instrumental resolution
        self.wave_i = np.linspace(np.min(self.wave), np.max(self.wave), self.wave.size)
        self.flux_i = interp1d(self.wave, self.flux, kind='linear', bounds_error=False)(self.wave_i)
        self.bflux = pyasl.instrBroadGaussFast(self.wave_i, self.flux_i, resolution, 
                                          edgeHandling="firstlast", equid=True)
        
        if vsini > 0.:
            self.vsini = vsini
            self.bflux = pyasl.fastRotBroad(self.wave_i, self.bflux, vsini=vsini, epsilon=0.)
            
        if in_place:
            setattr(self, 'flux', self.bflux)
            setattr(self, 'wave', self.wave_i)
        else:
            # interpolate back to original wavelength grid
            self.bflux = interp1d(self.wave_i, self.bflux, kind='linear', bounds_error=False)(self.wave)
        return self
    
    def get_telluric(self):
        # read Molecfit-generated telluric template
        file = '/home/dario/phd/lhs3844/espresso/templates/TELLURIC_ESPRESSO.fits'
        tdata = fits.getdata(file, 1)
        self.wave = tdata['mlambda'] * 1e4 # TELLURIC FRAME (vacuum)
        self.flux = tdata['mtrans']
        return self
    
    def get_star(self, T_eff=3036, log_g=5.06, cache=True, vacuum=False):
        from expecto import get_spectrum
        star = get_spectrum(T_eff=3036, log_g=5.06, cache=True, vacuum=vacuum)
        self.wave = star.spectral_axis.value
        self.flux = star.flux.value
        return self
    
    def blackbody(self, T,wave):
        ''' Returns the Planck function :math:`B_{\\nu}(T)` in units of
        :math:`\\rm erg/s/cm^2/Hz/steradian`.
        Args:
            T (float):
                Temperature in K.
            wave (numpy.ndarray):
                array containing the wavelength in AA.
        '''
        import scipy.constants as snc
        # Natural constants in CGS units
        c = snc.c * 1e2
        h = snc.h * 1e7
        kB = snc.k * 1e7
        wave_cm = wave * 1e-8
        nu = c / wave_cm
        planck = 2.*h*nu**3./c**2.
        planck /= (np.exp(h*nu/kB/T)-1.)
        return planck
    
    def gaussian_filter(self, window=15., mode='divide', debug=False):
        from scipy import ndimage
        lowpass = ndimage.gaussian_filter1d(self.flux, sigma=window)
        
        if debug:
            ax = plt.gca()
            ax.plot(self.wave, self.flux, label='flux')
            ax.plot(self.wave, lowpass, label='lowpass', lw=2., alpha=0.8)
            ax.legend()
            plt.show()
            
        self.flux = getattr(np, mode)(self.flux, lowpass)
        return self
    
    
    
    
    
if __name__ == '__main__':
    size = 4000
    mx = np.linspace(0, size*0.1, size)
    my = np.sin(0.1*mx+0.1)
    
    temp = Template(mx, my)
    print('Template size = ' , temp.flux.shape)
    temp.plot(lw=2.); plt.show()