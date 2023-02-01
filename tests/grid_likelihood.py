import numpy as np
from jax import random, vmap
import matplotlib.pyplot as plt
# import np
# from scipy.interpolate import interp1d, splev, splrep

def linear(x, theta):
    a, b = theta
    return theta[0] * x + theta[1]

def gaussian(x, theta):
    mu, sigma = theta
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# key = random.PRNGKey(42)
a_0, da = 3.0, 0.01
b_0, db = 0.2, 0.01
eps = 1e-4

x = np.linspace(2., 4., 300)


# a_true = a_0 + sigma * random.normal(key, shape=(x.shape))
# b_true = b_0 + sigma  * random.normal(key, shape=(x.shape))

a_true = np.random.normal(a_0, da, x.shape)
b_true = np.random.normal(b_0, db, x.shape)

theta_true = (a_true, b_true)


# noise_obs = sigma * random.normal(key, shape=(x.shape))
noise_obs = eps * np.random.uniform(1, 4., x.shape)
y_true = gaussian(x, theta_true)
y_obs = np.random.normal(y_true, noise_obs)

plot_data = True
if plot_data:
    plt.plot(x, y_obs, 'og')
    plt.plot(x, y_true,'-b')
    plt.show()


def get_prior(N_grid, a1=0.5, a2=1.5):
    a_grid = np.linspace(a1, a2, N_grid)
    a_middle = (a_grid[1:] + a_grid[:-1]) / 2
    a_step = (a_grid[1:] - a_grid[:-1])
    prior_density = 1 / (a_grid[-1] - a_grid[0])
    prior = prior_density * a_step
    return a_middle, prior, a_step

N_grid = 301
a_middle, a_prior, a_step = get_prior(N_grid, 1., 5.)
b_middle, b_prior, b_step = get_prior(N_grid//10, 0.1, 1.)

plot_priors = False
if plot_priors:
    plt.hist(a_middle, histtype='step', label="Flat prior on $a$", density=True)
    plt.hist(b_middle, histtype='step', label="Flat prior on $b$", density=True)

    plt.xlabel("$a$")
    plt.ylabel("Probability density")
    plt.legend(loc='best')
    plt.show()

def loglikelihood(a, b):
    # y = np.interp(linear(x, (a,b)), x, model)
    # y = splev(linear(x, (a,b)), cs, der=0)
    y_model = gaussian(x, (a,b))
    chi2 = np.power((y_obs - y_model) / noise_obs, 2).sum()
    return -chi2 / 2

grid = np.meshgrid(a_middle, b_middle)
grid_cellsize = np.product(np.meshgrid(a_step, b_step), axis=0)

# grid_posterior = np.array([loglikelihood((a,b)) for a in a_middle])
# log_like_b = vmap(loglikelihood, in_axes=(None,0))
# log_like_a = vmap(log_like_b, in_axes=(0,None))
# grid_posterior = log_like_a(a_middle, b_middle)
grid_unnormalised_logposterior = np.vectorize(loglikelihood)(grid[0], grid[1]) + np.log(grid_cellsize)
grid_unnormalised_posterior = np.exp(grid_unnormalised_logposterior)
grid_logevidence = np.log(grid_unnormalised_posterior.sum())
grid_posterior = np.exp(grid_unnormalised_logposterior - grid_logevidence)
print(grid_posterior.sum())

# plt.plot(a_middle, grid_posterior, drawstyle='steps-mid')

# plt.ylim(-1000, 0);
plt.ylabel("log-likelihood")
plt.xlabel("$\\alpha$")
plt.show()

plt.subplot(2, 2, 3)
plt.imshow(
      grid_unnormalised_logposterior[::-1],
      extent=(a_middle[0], a_middle[-1], b_middle[0], b_middle[-1]),
      aspect='auto', cmap='gray_r')
#plt.colorbar(label='log-likelihood')
plt.xlabel("$\\alpha$")
plt.ylabel("$\\beta$");
plt.subplot(2, 2, 1)
plt.plot(a_middle, grid_unnormalised_logposterior.sum(axis=0))
plt.xlabel("$\\alpha$")
plt.yticks([])
plt.subplot(2, 2, 4)
plt.plot(grid_unnormalised_logposterior.sum(axis=1), b_middle)
plt.ylabel("$\\beta$")
plt.xticks([])
plt.suptitle("Marginal and conditional probability distributions");
plt.show()

