"""Small numerical helpers used by the entrepreneurship example."""

import numpy as np


def nonlinspace(x_min, x_max, n, phi=1.2):
    """Grid with extra points close to the borrowing constraint."""
    return x_min + (x_max-x_min)*np.linspace(0.0, 1.0, n)**phi


def lognormal_gauss_hermite(sigma, n):
    """Mean-one lognormal shocks; sigma is the standard deviation in levels."""
    if sigma == 0 or n == 1: return np.ones(1), np.ones(1)
    x, w = np.polynomial.hermite.hermgauss(n)
    s = np.sqrt(np.log1p(sigma**2))
    shocks = np.exp(np.sqrt(2)*s*x-s**2/2)
    w = w/np.sqrt(np.pi)
    return shocks/np.sum(w*shocks), w


def interp(x, y, x_new):
    """Linear interpolation with linear extrapolation."""
    x_new = np.asarray(x_new); flat = x_new.ravel(); out = np.interp(flat, x, y)
    left, right = flat < x[0], flat > x[-1]
    out[left] = y[0]+(flat[left]-x[0])*(y[1]-y[0])/(x[1]-x[0])
    out[right] = y[-1]+(flat[right]-x[-1])*(y[-1]-y[-2])/(x[-1]-x[-2])
    return out.reshape(x_new.shape)


def logsum(v, sigma):
    """Inclusive value and logit probabilities; sigma=0 gives an exact max."""
    vmax = np.max(v, axis=0)
    if sigma > 1e-12:
        ev = np.exp((v-vmax)/sigma); prob = ev/np.sum(ev, axis=0)
        return vmax+sigma*np.log(np.sum(ev, axis=0)), prob
    choice = np.argmax(v, axis=0); prob = np.zeros_like(v)
    np.put_along_axis(prob, choice[None, :], 1.0, axis=0)
    return vmax, prob
