import numpy as numpy
import jax.numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from scipy import ndimage

class Align:
    def __init__(self, dco, RVt=np.arange(-10, 10.2, 0.1)):
        self.dco = dco.copy().normalise()
        self.dco_corr = dco.copy() # for the output

        self.RVt = RVt
        self.pixels = np.arange(0, self.dco.nPix)

        # default settings
        self.shifts = numpy.zeros(self.dco.nObs)
        self.ccf = numpy.zeros((self.dco.nObs, self.RVt.size))
        self.window = 30. # pixels, for lowpass smoothing

        # Remove continuum
        lowpass = ndimage.gaussian_filter(self.dco.flux, [0, self.window])
        self.dco.flux = self.dco.flux / lowpass


    @staticmethod
    def xcorr(f,g):
        nx = len(f)
        R = np.dot(f,g)/nx
        varf = np.dot(f,f)/nx
        varg = np.dot(g,g)/nx

        CC = R / np.sqrt(varf*varg)
        return CC

    @staticmethod
    def gaussian(x, a, x0, sigma, y0):
        return y0 + a*np.exp(-(x-x0)**2/(2*sigma**2))

    @staticmethod
    def clip(f, sigma=3.):
        mean, std = np.mean(f), np.std(f)
        mask = np.abs(f - mean) > std*sigma
        f = f.at[mask].set(np.nan)
        return f
    def get_shift(self, j, ax=None):
        # edge = 100 # ignore the first/last N pixels
        f, g = self.dco.flux[0], self.dco.flux[j]
        f = self.clip(f)
        g = self.clip(g)
        nans = np.isnan(f)+np.isnan(g)
        # print(nans[nans==True].size)
        # fw, gw = self.dco.wlt[0, edge:-edge], self.dco.wlt[j]
        # fw = self.pixels[edge:-edge]
        fw = self.pixels[~nans]

        # beta = 1 - (self.RVt/c)
        cs = splrep(self.pixels[~nans], g[~nans])
        # self.ccf = np.array([self.xcorr(f, splev(fw*b, cs)) for b in beta])
        # vfun = vmap(self.xcorr, in_axes=(None,0), out_axes=0)
        self.ccf[j] = np.array([self.xcorr(f[~nans], splev(fw+s, cs)) for s in self.RVt])
        # self.ccf[j] = self.xcorr(f[~nans], splev(fw+self.RVt, cs))

        if not ax is None:
            args = dict(alpha=0.5, ms=1.)
            ax.plot(self.RVt, self.ccf[j], '--o', label='Frame {:}'.format(j), **args)
            # ax.plot(self.pixels[~nans], f[~nans], label='f')
            # ax.plot(self.pixels[~nans], g[~nans], label='g')
        return self

    def run(self, ax=None):
        [self.get_shift(j, ax=ax) for j in range(self.dco.nObs)]
        self.shifts = self.RVt[self.ccf.argmax(axis=1)]
        return self


    def apply_shifts(self):
        self.run()
        for j in range(1,self.dco.nObs):
            cs = splrep(self.pixels, self.dco_corr.flux[j,])
            self.dco_corr.flux[j,] = splev(self.pixels + self.shifts[j], cs)
        return self

    def plot_results(self, outname=None):
        fig, ax = plt.subplots(1,2, figsize=(12,4))

        cmap = plt.cm.get_cmap('viridis', self.dco.nObs)
        colors = np.array([cmap(x) for x in range(self.dco.nObs)])
        for j in range(self.dco.nObs):
            ax[0].plot(self.RVt, self.ccf[j], '--o', ms=1., color=colors[j], alpha=0.35)
            ax[1].plot(j, self.shifts[j], 'o', color=colors[j], ms=5.)


        ax[1].plot(self.shifts, '--k', alpha=0.4)


        ax[0].set(xlabel='Pixel shift', ylabel='CCF')

        ylim = np.abs(ax[1].get_ylim()).max()
        ax[1].set(xlabel='Frame number', ylabel='Pixel drift', ylim=(-ylim, ylim))

        # ax[1].legend()
        plt.show()
        if outname != None:
            fig.savefig(outname)
        return None