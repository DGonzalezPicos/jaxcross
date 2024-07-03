import jax.numpy as np
from jaxcross import CRIRES, Template, CCF, Planet, KpV
import numpy
import matplotlib.pyplot as plt
import pathlib

from scipy.interpolate import interp1d

def get_mask(dv, kp):
    test_planet = planet.copy()
    test_planet.Kp = kp
    mask = numpy.zeros_like(ccf.map)
    for i in range(len(test_planet.RV)):
        shift = test_planet.RV[i] + dv
        j = numpy.argmin(numpy.abs(ccf.RV - shift))
        # g[i,j] = 1.
        mask[i,j] = 1.
    return mask.astype(bool)

# Load planet
path = pathlib.Path("/home/dario/phd/pycrires/pycrires/product/obs_staring/")
files = sorted(path.glob("cr2res_obs_staring_extracted_*.fits"))
planet = Planet('MASCARA-1b').read_header(files)

ccf = CCF().load('ccf_co_inj.npy')

fig, ax = plt.subplots(1,2,figsize=(12,4))
ccf.phase = planet.phase
obj = ccf.imshow(ax=ax[0])

# vmin, vmax = obj.get_clim()
# mask = get_mask(0., 200.4)
# ccf_masked = numpy.where(mask, ccf.map, 0.)
# ax[1].imshow(ccf_masked, origin='lower', aspect='auto', 
#              extent=[ccf.RV[0], ccf.RV[-1], ccf.phase[0], ccf.phase[-1]],
#              vmin=vmin, vmax=vmax)
# plt.show()

def get_kpv(dv, kp):
    mask = get_mask(dv, kp)
    ccf_masked = numpy.where(mask, ccf.map, 0.)
    return numpy.sum(ccf_masked)

kpvec = numpy.arange(-10, 10.,2.) + planet.Kp
dvvec = numpy.arange(-20, 20., 2.)
kpv = numpy.zeros((len(kpvec), len(dvvec)))
for i, kp in enumerate(kpvec):
    for j,dv in enumerate(dvvec):
        print(i,j)
        kpv[i,j] = get_kpv(dv, kp)
        
ax[1].imshow(kpv, origin='lower', aspect='auto', extent=[dvvec[0], dvvec[-1], kpvec[0], kpvec[-1]])
plt.show()
# kpv[i,j] = get_kpv(dv, kp)


    
    
    
    
# g = numpy.zeros_like(ccf.map)



# ax[1].imshow(kpv, origin='lower', aspect='auto', extent=[dvvec[0], dvvec[-1], kpvec[0], kpvec[-1]])
    
# ax[1].plot(planet.RV+10., planet.phase, 'or')
# ax[1].imshow(g, origin='lower', aspect='auto', extent=[ccf.RV[0], ccf.RV[-1], ccf.phase[0], ccf.phase[-1]],
#                 alpha=0.4)
# plt.show()

# kpv = KpV(ccf, planet, deltaRV=1., kp_radius=30)
# kpv.run(ignore_eclipse=False)
# kpv.fancy_figure()
# plt.show()
# dv = np.arange(-30, 30, 1)

# kpvec = np.arange(-10, 10, 1) + planet.Kp
# new_ccf = np.zeros((len(kpvec), len(dv)))
# for ikp in range(kpvec):
#     new_planet = planet.copy()
#     new_planet.Kp = kpvec[ikp]
# rv_shift = planet.RV[:,np.newaxis] + dv[np.newaxis,:] # (N, M)
# new_ccf = np.sum(interp1d(ccf.RV, ccf.map[0], kind='linear', fill_value=0.0)(rv_shift), axis=0)
