import numpy as np
from astropy.time import Time
import astropy.constants as const
from astropy import units as u, coordinates as coord
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
import copy
c = const.c.to('km/s').value
class Planet:
    
    def __init__(self, name, file=None, header=None):
        self.name = name
        
        if name == 'MASCARA-1b':
            self.read('/home/dario/phd/mascara1/mascara1b.dat')
        
        if file is not None:
            self.read(file)
        if header is not None:
            self.__set_header(header)
            
        self._frame = 'telluric' # default
        
        self.observatory = 'paranal' # default (observatory)
        # astropy location object (EarthLocation.get_site_names() for full list)
        loc_warning = "Observatory not found in EarthLocation.get_site_names()."
        assert self.observatory in EarthLocation.get_site_names(), loc_warning
        self.location = EarthLocation.of_site(self.observatory)

            
        
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
        """Read RA, DEC, airmass, DATE-OBS from FITS header."""
        
        hdr_keys = ['ra','dec']
        air_key = ['HIERARCH ESO TEL AIRM '+i for i in ['START','END']]
        airmass, date = ([] for _ in range(2))
        for f in files:
            with fits.open(f) as hdul:
                # hdul.info()
                header = hdul[0].header
            airmass.append(np.mean([header[key] for key in air_key]))
            # MJD.append(header['MJD-OBS'])
            date.append(header['DATE-OBS'])
        my_header = {key:header[key] for key in hdr_keys}
        my_header['airmass'] = np.round(airmass, 4)
        # my_header['MJD'] = MJD
        my_header['date'] = date
        self.set_header(my_header) # save header
        return self
    
    def set_header(self, header={}):
        for k,v in header.items():
            setattr(self, k, v)
        self.time = Time(self.date, format='isot', scale='tdb')
        
        if hasattr(self, 'T_0'): self.Tc = Time(self.T_0, format='jd',scale='tdb') 
        if hasattr(self, 'T_14'): self.T_14 /= 24. # from hours to days
        if hasattr(self, 'T_12'): self.T_12 /= (60*24.) # from minutes to days

        if hasattr(self, 'a'):
            self.v_orb = (2*np.pi*self.a*u.AU / (self.P*u.d)).to(u.km/u.s)
            self.Kp = (self.v_orb * np.sin(np.radians(self.i))).value
            
        if hasattr(self, 'ra') and hasattr(self, 'dec'): # units must be in DEGREES
            self.sky_coord= SkyCoord(self.ra, self.dec, unit=(u.deg, u.deg))
            
        self.dphi = 0.0 # default (phase shift)

        return self
    
    @property
    def frame(self): 
        return self._frame
    @frame.setter
    def frame(self, frame):
        assert frame in ['barycentric', 'stellar','telluric','planet'], "Frame not recognized."
        self._frame = frame
        
    
    @property
    def JD(self):
        return self.time.jd
    
    @property
    def MJD(self):
        return self.time.mjd
    
    def get_berv(self):
        from barycorrpy import get_BC_vel 
        obsname = "paranal"
        if hasattr(self, 'observatory'):
            obsname = self.observatory
        vcorr,_,_ = get_BC_vel(Time(self.JD, format='jd'), ra=self.ra, dec=self.dec, obsname=obsname)
        self.berv = vcorr / 1e3 # m/s to km/s 
        return self
        
    @property
    def BJD(self):
        '''convert MJD to BJD'''
        #Convert MJD to BJD to account for light travel time. Adopted from Astropy manual.
        t = Time(self.MJD, format='mjd',scale='tdb',location=self.location) 
        ltt_bary = t.light_travel_time(self.sky_coord)
        return t.tdb + ltt_bary # = BJD  
    
    @property
    def phase(self):
        return ((self.BJD-self.Tc).value % self.P) / self.P
    
    @property
    def RV(self):
        if not hasattr(self, 'berv'):
            self.get_berv()
        #Derive the instantaneous radial velocity at which the planet is expected to be.
        RV_planet = np.sin(2*np.pi*self.phase)*self.Kp
        if self.frame == 'stellar':
            rvel = 0.0
        elif self.frame == 'telluric':
            rvel = ((self.v_sys*u.km/u.s)-self.berv*u.km/u.s).value
            
        elif self.frame == 'barycentric':
            rvel = self.v_sys # When `v_sys` is an array --> v_sys(t) = v_sys + RV_star(t)
            # WARNING: the sign of v_sys here is (+) for ESPRESSO since it already has the correct sign
            # for LHS3844: RV_sys = -10.54 km/s (from the header)
        elif self.frame == 'planet':
            return np.zeros_like(self.berv)
            
        # print(rvel)
        return (rvel + RV_planet)
    
    @property
    def cphase(self):
        cont_phase = np.copy(self.phase)
        cont_phase[cont_phase>0.50] -= 1.0 # continuous phase
        return cont_phase
    @property
    def transit_mask(self):
        assert hasattr(self, 'T_14'), "T_14 not defined."
        self.phase_14 = 0.50 * self.T_14 / self.P # phase duration of transit
        self.phase_12 = self.T_12 / self.P # phase duration of ingress/egress
        ## Include exposures that are partially in-transit ## 
        # Take the mean of the array of exposure times (should be constant)
        # the factor "2" is optimistic (account for entrance and exit uncertainty)
        dphase = np.mean(self.exptime) / (3600. * 24. * self.P) # uncertainty in phase
        return np.abs(self.cphase) < (self.phase_14+dphase) # frames IN-eclipse
    @property
    def in_transit(self):
        return self.transit_mask.sum()
    @property
    def out_transit(self):
        return np.sum(~self.transit_mask)
        
    def mask_eclipse(self, return_mask=False, debug=False):
        '''given the duration of eclipse `t_14` in days
        return the PLANET with the frames masked'''
        
        shape_in = self.RV.size
        phase = self.phase
        self.phase_14 = 0.5 * ((self.T_14) % self.P) / self.P
        
        mask = np.abs(phase - 0.50) < self.phase_14 # frames IN-eclipse
            
        if return_mask:
            return mask

        # Update planet header with the MASKED vectors
        for key in ['time','airmass']:
            setattr(self, key, getattr(self, key)[~mask])
        if hasattr(self, 'berv'):
            self.berv = self.berv[~mask]
            
        if debug:
            print('Original self.shape = {:}'.format(shape_in))
            print('After ECLIPSE masking = {:}'.format(self.RV.size))
        return self
    
    def copy(self):
        return copy.deepcopy(self)
    
    def trail(self, ax, wave0=None, frame=None, **kwargs):
        pl_copy = self.copy()
        pl_copy.frame = frame or self.frame
        if wave0 is not None:
            x = wave0 * (1 + pl_copy.RV[self.transit_mask] / c)
        else:
            x = pl_copy.RV[self.transit_mask]
            
        [ax.plot(x+i, pl_copy.cphase[self.transit_mask], **kwargs) for i in (-1,1)]
    

    
    # def set_header(self, header):
    #     for k,v in header.items():
    #         setattr(self, k, v)
    #     if hasattr(self, 'date'):
    #         self.time = Time(self.date, format='isot', scale='tdb')
    #     return self
            
if __name__ == '__main__':
    from astropy.io import fits
    import pathlib

    planet_file = '/home/dario/phd/mascara1/mascara1b.dat' 

    path = pathlib.Path("/home/dario/phd/pycrires/pycrires/product/obs_staring/")
    files = sorted(path.glob("cr2res_obs_staring_extracted_*.fits"))
    
    def get_header(files):
        """Read RA, DEC, airmass, DATE-OBS from FITS header."""
        
        hdr_keys = ['ra','dec']
        air_key = ['HIERARCH ESO TEL AIRM '+i for i in ['START','END']]
        airmass, date = ([] for _ in range(2))
        for f in files:
            with fits.open(f) as hdul:
                # hdul.info()
                header = hdul[0].header
            airmass.append(np.mean([header[key] for key in air_key]))
            # MJD.append(header['MJD-OBS'])
            date.append(header['DATE-OBS'])
        my_header = {key:header[key] for key in hdr_keys}
        my_header['airmass'] = np.round(airmass, 4)
        # my_header['MJD'] = MJD
        my_header['DATE'] = date
        return my_header
    
    # Read header values from individual FITS files and save them as a dictionary
    my_header = get_header(files)
    
    planet = Planet('MASCARA-1b').read_header(files)
    print(planet)
    print(planet.phase)