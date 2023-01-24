import numpy
from astropy.time import Time
import astropy.constants as const
from astropy import units as u, coordinates as coord
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits

class Planet:
    
    def __init__(self, name, header=None):
        self.name = name
        
        if name == 'MASCARA-1b':
            self.read('/home/dario/phd/mascara1/mascara1b.dat')
        
        if header is not None:
            self.__set_header(header)
            
        
    def __str__(self):
        row0 = '\n'+("".join("-" for _ in range(60))) + '\n'
        row1 = f"\t \t Planet {self.name}\n"
        row2 = ("".join("-" for _ in range(60))) + '\n'
        row3 = '\n'.join([f'{k:5} = {getattr(self,k):14} <---> {v}' for k,v in self.info.items()])
        return f"{row0}{row1}{row2}{row3}"
    
    def __repr__(self):
        return self.__str__()
    
    def read(self, filename):
        with open(filename) as f:
            rows = f.readlines()
            data = [float(x.split('#')[0]) for x in rows] # read data
            header = [str(x.split('#')[1][:-1]) for x in rows] # read header
            keys = [x.split(':')[0].lstrip() for x in header] # select keys from header
            info = [x.split(':')[1].lstrip() for x in header] # select additional info from header
        for key, value in zip(keys, data):
            setattr(self, key, value)
        # Store additional info in a dictionary (units, description, source, etc.)
        info_dict = {k:v for k,v in zip(keys, info)}
        setattr(self, 'info', info_dict)

        return self
    
    def read_header(self, files):
        """Read RA, DEC, AIRMASS, DATE-OBS from FITS header."""
        
        hdr_keys = ['RA','DEC']
        air_key = ['HIERARCH ESO TEL AIRM '+i for i in ['START','END']]
        airmass, date = ([] for _ in range(2))
        for f in files:
            with fits.open(f) as hdul:
                # hdul.info()
                header = hdul[0].header
            airmass.append(numpy.mean([header[key] for key in air_key]))
            # MJD.append(header['MJD-OBS'])
            date.append(header['DATE-OBS'])
        my_header = {key:header[key] for key in hdr_keys}
        my_header['AIRMASS'] = numpy.round(airmass, 4)
        # my_header['MJD'] = MJD
        my_header['DATE'] = date
        self.__set_header(my_header) # save header
        return self
    
    def __set_header(self, header):
        for k,v in header.items():
            setattr(self, k, v)
        self.time = Time(self.DATE, format='isot', scale='tdb')
        
        if hasattr(self, 'T_0'): self.Tc = Time(self.T_0, format='jd',scale='tdb') 
        if hasattr(self, 'T_14'): self.T_14 /= 24. # from hours to days

        if hasattr(self, 'a'):
            self.v_orb = (2*numpy.pi*self.a*u.AU / (self.P*u.d)).to(u.km/u.s)
            self.Kp = (self.v_orb * numpy.sin(numpy.radians(self.i))).value
            
        if hasattr(self, 'RA') and hasattr(self, 'DEC'): # units must be in DEGREES
            self.sky_coord= SkyCoord(self.RA, self.DEC, unit=(u.deg, u.deg))
            
        self.frame = 'telluric' # default
        self.dphi = 0.0 # default (phase shift)
        self.observatory = 'paranal' # default (observatory)
        
        # astropy location object (EarthLocation.get_site_names() for full list)
        loc_warning = "Observatory not found in EarthLocation.get_site_names()."
        assert self.observatory in EarthLocation.get_site_names(), loc_warning
        self.location = EarthLocation.of_site(self.observatory)
        return self
    
    @property
    def jd(self):
        return self.time.jd
    
    @property
    def mjd(self):
        return self.time.mjd
    
    def get_BERV(self):
        from barycorrpy import get_BC_vel 
        obsname = "paranal"
        if hasattr(self, 'observatory'):
            obsname = self.observatory
        vcorr,_,_ = get_BC_vel(Time(self.jd, format='jd'), ra=self.RA, dec=self.DEC, obsname=obsname)
        self.BERV = vcorr / 1e3 # m/s to km/s 
        return self
        
    @property
    def BJD(self):
        '''convert MJD to BJD'''
        #Convert MJD to BJD to account for light travel time. Adopted from Astropy manual.
        t = Time(self.mjd, format='mjd',scale='tdb',location=self.location) 
        ltt_bary = t.light_travel_time(self.sky_coord)
        return t.tdb + ltt_bary # = BJD  
    
    @property
    def phase(self):
        return ((self.BJD-self.Tc).value % self.P) / self.P
    
    @property
    def RV(self):
        if not hasattr(self, 'BERV'):
            self.get_BERV()
        #Derive the instantaneous radial velocity at which the planet is expected to be.
        RV_planet = numpy.sin(2*numpy.pi*self.phase)*self.Kp
        if self.frame == 'stellar':
            rvel = 0.0
        elif self.frame == 'telluric':
            rvel = ((self.v_sys*u.km/u.s)-self.BERV*u.km/u.s).value
            
        elif self.frame == 'barycentric':
            rvel = self.v_sys
            
        elif self.frame == 'planet':
            return numpy.zeros_like(self.BERV)
            
        # print(rvel)
        return (rvel + RV_planet)
            
if __name__ == '__main__':
    from astropy.io import fits
    import pathlib

    planet_file = '/home/dario/phd/mascara1/mascara1b.dat' 

    path = pathlib.Path("/home/dario/phd/pycrires/pycrires/product/obs_staring/")
    files = sorted(path.glob("cr2res_obs_staring_extracted_*.fits"))
    
    def get_header(files):
        """Read RA, DEC, AIRMASS, DATE-OBS from FITS header."""
        
        hdr_keys = ['RA','DEC']
        air_key = ['HIERARCH ESO TEL AIRM '+i for i in ['START','END']]
        airmass, date = ([] for _ in range(2))
        for f in files:
            with fits.open(f) as hdul:
                # hdul.info()
                header = hdul[0].header
            airmass.append(numpy.mean([header[key] for key in air_key]))
            # MJD.append(header['MJD-OBS'])
            date.append(header['DATE-OBS'])
        my_header = {key:header[key] for key in hdr_keys}
        my_header['AIRMASS'] = numpy.round(airmass, 4)
        # my_header['MJD'] = MJD
        my_header['DATE'] = date
        return my_header
    
    # Read header values from individual FITS files and save them as a dictionary
    my_header = get_header(files)
    
    planet = Planet('MASCARA-1b').read_header(files)
    print(planet)
    print(planet.phase)