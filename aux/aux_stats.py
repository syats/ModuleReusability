import numpy as np
from scipy import integrate
from scipy import optimize, stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import argrelextrema


# This module aims at computing the bimodality of a distribution. It assumes that the distribution
# is a mixture model of two gaussians.

def truncated_gaussian_lower(x, mu, sigma, A):
    return np.clip(A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2), a_min=0, a_max=None)


def truncated_gaussian_upper(x, mu, sigma, A):
    return np.clip(A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2), a_min=None, a_max=1)


def m_mixture_model(x, mu1, sigma1, A1, mu2, sigma2, A2, m=1):
    # res = np.zeros_like(x)
    # if (x < 0) or (x > 1):
    #     return 0
    res = truncated_gaussian_lower(x, mu1, sigma1, A1) + truncated_gaussian_upper(x, mu2, sigma2, A2)
    res[x < 0] = 0
    res[x > m] = 0
    return res


def find_two_maxima(u, ax=None, ss=0.001):
    if ax is None:
        fig, ax = plt.subplots()

    kde = stats.gaussian_kde(u)
    ar = np.arange(0, 1, ss)
    kde_eva = kde(ar)
    extrema = argrelextrema(kde_eva, np.greater)
    if len(extrema[0]) == 1:
        m1 = extrema[0][0]*ss
        m2 = extrema[0][0]*ss
    else:
        m1, m2 = extrema[0][0]*ss, extrema[0][1]*ss

    if ax != 0:
        plot_hist_and_fit(ax, u, m1, m2)
    return m1, m2


# U is a bi-modaly distributed set of points  in [0,1]
# r is a first estimate of where the first mode is distributed
def fit_mixture_to_data(u, r=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    kde = stats.gaussian_kde(u)
    if r == None:
        ar = np.arange(0,1,0.001)
        r = np.argmax(kde(ar))*0.001
    n = len(u)
    sum_prob = integrate.quad(kde, 0, 1)[0]
    y = kde(u) / sum_prob
    # Estimates: mu  sigma  A     mu     sigma    A
    estimates = [r,   0.4,  20,  max(u)*0.9,   0.5,   1]
    low_bound = [0,   0,    0,     0,       0,    0]
    upp_bound = [1,   0.9  ,np.inf,  max(u),       0.5,    np.inf]
    mixture_model = lambda x1, x2, x3, x4, x5, x6, x7: m_mixture_model(x1, x2, x3, x4, x5, x6, x7, m=1)

    params, cov = optimize.curve_fit(f=mixture_model,
                                     xdata=u, ydata=y,
                                     p0=estimates, bounds=(low_bound, upp_bound))


    m1, m2 = params[0], params[3]
    y3 = mixture_model(np.arange(0, 1, 0.01), m1, params[1], params[2],
                       m2, params[4], params[5])
    y3 = y3 / y3.sum()

    if ax!= 0:
        plot_hist_and_fit(ax, u, m1, m2)
    return m1, m2

def plot_hist_and_fit(ax,u,m1,m2):

    with plt.style.context("seaborn-white"):

        sns.kdeplot(u, color="black", ax=ax)
        ax.axvline(m1, linestyle=":", color="red")
        ax.axvline(m2, linestyle=":", color="green")
        #plt.plot(np.arange(0, m, 0.01), y3,
        #         color="green")  # The red line is now your custom pdf with area-under-curve = 0.998 in the domain..
        plt.hist(u,20)


