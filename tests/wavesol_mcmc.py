import jax.numpy as np
import matplotlib.pyplot as plt
import numpy
from jaxcross import CRIRES, Template, CCF
import emcee 
from scipy.interpolate import interp1d

filename = "/home/dario/phd/pycrires/pycrires/calib/run_skycalc/transm_spec.dat"
mx, my = numpy.loadtxt(filename, unpack=True)
template = Template(mx, my).crop(2000., 2100.)
# template.plot()

# template.shift(RV=50.)
# template.plot(ls='--')
# plt.show()

def log_likelihood(theta, x, y, temp_inter):
    x1,x2,x3 = theta
    # x_s = x[100:-100] * (1 + RV/299792.458)
    xt = x[100:-100]
    x_s = (xt-x1) * (xt-x2) * (xt-x3)
    model = temp_inter(x_s)
    return -0.5 * numpy.sum((y[100:-100] - model) ** 2)


def log_prior(theta, x):
    x1,x2,x3 = theta
    p33 = numpy.percentile(x, 33)
    p66 = numpy.percentile(x, 66)
    if -x.min() < x1 < p33 and p33 < x2 < p66 and p66 < x3 < x.max():
        return 0.0
    return -np.inf

def log_probability(theta, x, y, temp_inter):
    lp = log_prior(theta,x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, temp_inter)

pos = 0.0 + 1e-1 * numpy.random.randn(32,3)
nwalkers, ndim = pos.shape
dRV = 3.3
x = template.wave * (1 - dRV/299792.458)
noise = 0.05 * numpy.random.randn(len(x))
y = template.flux + noise
temp_inter = interp1d(template.wave, template.flux, kind='linear', 
                      bounds_error=False, fill_value=0.)
plt.plot(x, y, label='Data')
plt.plot(template.wave, template.flux, label='Template')
plt.legend()
plt.show()
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(x,y, temp_inter)
)
sampler.run_mcmc(pos, 5000, progress=True);

# fig, ax = plt.subplots(1, figsize=(10, 7), sharex=True)
# samples = sampler.get_chain()
labels = ["x1", "x2", "x3"]
# ax.plot(samples[:, :,], "k", alpha=0.3)
# ax.set_xlim(0, len(samples))
# ax.set_ylabel(labels[i])
# ax.yaxis.set_label_coords(-0.1, 0.5)

# axes[-1].set_xlabel("step number");

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
print(flat_samples.shape)

import corner

fig = corner.corner(
    flat_samples, labels=labels)
plt.show()