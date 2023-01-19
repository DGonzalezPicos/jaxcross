import jax.numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


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
    
if __name__ == '__main__':
    size = 4000
    mx = np.linspace(0, size*0.1, size)
    my = np.sin(0.1*mx+0.1)
    
    temp = Template(mx, my)
    print('Template size = ' , temp.flux.shape)
    temp.plot(lw=2.); plt.show()