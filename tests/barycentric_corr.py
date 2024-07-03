import numpy
from astropy.time import Time

from barycorrpy import get_BC_vel 
from jaxcross import Planet


w189_file = '/home/dario/AstronomyLeiden/MRP/wasp189/harpsn/data/night1/planet.npy'
w189_dict = numpy.load(w189_file, allow_pickle=True).tolist()
w189 = Planet('WASP-189b')
for key in w189_dict.keys():
    setattr(w189, key, w189_dict[key])
    
vcorr,_,_ = get_BC_vel(Time(w189.MJD, format='mjd'), ra=w189.RA_DEG, dec=w189.DEC_DEG, obsname='lapalma')

# Good agreement between BERV and barycorrpy within 0.01 km/s
# BERV is the barycentric velocity given by the HARPS-N DRS pipeline
print(numpy.isclose(vcorr/1e3, w189.BERV, atol=1e-2))