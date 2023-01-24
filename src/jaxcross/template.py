import jax.numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class Template:
    
    def __init__(self, wave, flux):
        self.wave = wave
        self.flux = flux
        
        self.wave_unit = 'nm' # TO-DO: implement auto-setting of units...
    def read(self, fits_file, debug=True):
        with fits.open(fits_file) as hdul:
            if debug: hdul.info()
            self.wave = hdul[1].data
            self.flux = hdul[2].data
        return self
    
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
    
if __name__ == '__main__':
    size = 4000
    mx = np.linspace(0, size*0.1, size)
    my = np.sin(0.1*mx+0.1)
    
    temp = Template(mx, my)
    print('Template size = ' , temp.flux.shape)
    temp.plot(lw=2.); plt.show()