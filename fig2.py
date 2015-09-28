#!/usr/bin/env python2
# encoding: utf-8
'''
fig2.py

Written by:
    Omri Har-Shemesh, Computational Science Lab, University of Amsterdam
        O.HarShemesh@uva.nl

Last updated on 25 September 2015

Description:
    Figure 2 in Ref.[1]

References:
    [1] O. Har-Shemesh, R. Quax, B. Miñano, A.G. Hoekstra, P.M.A. Sloot, Non-parametric
        estimation of Fisher information from real data, arxiv:1507.00964[stat.CO]
Functions:

Dependencies:
    numpy
    matplotlib
    timeit
    cPickle
    os
    gzip
    npfi.py

'''
from __future__ import division

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

import os
import gzip
import cPickle as pickle
import timeit

from npfi import npfi, get_pdfs_from_data

def simulate_data(ss, N, rep, e, zero, G, alpha, fname):
    """ Simulates the data for the plot

    Args:
        ss: An array of sigma values to estimate the FI at.
        N: Number of data points for each PDF.
        rep: Number of repetitions of the whole simulation.
        e: The value of the epsilon parameter.
        zero: What should npfi consider as zero
        G: G for DEFT
        alpha: alpha for DEFT
        fname: Name of the file where the simulation data will be stored.

    Returns:
        data: A dictionary with all simulated data, which was also stored to
            the file.
    """
    # All list containers we need to store the values we compute
    FI_deft_median, FI_deft_5, FI_deft_95 = [], [], []
    FI_kde_median,  FI_kde_5,  FI_kde_95  = [], [], []
    err_deft_median, err_deft_5, err_deft_95 = [], [], []
    err_kde_median,  err_kde_5,  err_kde_95  = [], [], []
    FI_deft_values_all, FI_kde_values_all = [], []
    dss = []

    # Go over all sigma values in ss
    for i, s in enumerate(ss):
        real_FI = 2 / s ** 2
        ds = s / (e * np.sqrt(N))  # Choose ds according to desired epsilon
        # If ds >= s we have a problem of sampling with negative std
        while ds >= s:
            ds *= 0.9
        dss.append(ds)

        # Estimate the FI for rep repetitions
        FI_deft_values, FI_kde_values = [], []
        for j in range(rep):
            sim_data = [normal(size=N, scale=s),
                    normal(size=N, scale=s-ds),
                    normal(size=N, scale=s+ds)]
            pdfs_deft, bbox_deft = get_pdfs_from_data(sim_data, method="deft", G=G,
                                                      alpha=alpha, bbox="adjust")
            pdfs_kde, bbox_kde = get_pdfs_from_data(sim_data, method="gaussian_kde")
            FI_deft, a, b = npfi(pdfs_deft, ds, bounds=bbox_deft,
                                               logarithmic=False, zero=zero, N=N)
            FI_kde, a, b = npfi(pdfs_kde, ds, bounds=bbox_kde,
                                               logarithmic=True, zero=zero, N=N)
            FI_deft_values.append(FI_deft)
            FI_kde_values.append(FI_kde)

        # More convenient to use as numpy arrays
        FI_deft_values = np.array(FI_deft_values)
        FI_kde_values = np.array(FI_kde_values)

        FI_deft_values_all.append(FI_deft_values)
        FI_kde_values_all.append(FI_kde_values)

        # Compute statistics from the values we obtained
        FI_deft_median.append(np.median(FI_deft_values))
        FI_deft_5.append(np.percentile(FI_deft_values, 5))
        FI_deft_95.append(np.percentile(FI_deft_values, 95))
        FI_kde_median.append(np.median(FI_kde_values))
        FI_kde_5.append(np.percentile(FI_kde_values, 5))
        FI_kde_95.append(np.percentile(FI_kde_values, 95))

        # Compute relative error statistics
        err_deft_values = (FI_deft_values - real_FI) / real_FI
        err_deft_median.append(np.median(err_deft_values))
        err_deft_5.append(np.percentile(err_deft_values, 5))
        err_deft_95.append(np.percentile(err_deft_values, 95))
        err_kde_values = (FI_kde_values - real_FI) / real_FI
        err_kde_median.append(np.median(err_kde_values))
        err_kde_5.append(np.percentile(err_kde_values, 5))
        err_kde_95.append(np.percentile(err_kde_values, 95))

        if __debug__:
            print("Finished %d from %d values" % (i+1, len(ss)))

    f = gzip.open(fname, "wb")
    data = dict(ss=ss, dss=dss, FI_deft_values_all=FI_deft_values_all,
                FI_kde_values_all=FI_kde_values_all,
                FI_deft_median=FI_deft_median, FI_kde_median=FI_kde_median,
                FI_deft_5=FI_deft_5, FI_deft_95=FI_deft_95,
                FI_kde_5=FI_kde_5, FI_kde_95=FI_kde_95,
                err_deft_median=err_deft_median, err_kde_median=err_kde_median,
                err_deft_5=err_deft_5, err_deft_95=err_deft_95,
                err_kde_5=err_kde_5, err_kde_95=err_kde_95)
    pickle.dump(data, f)
    f.close()

    return data

def plot_data(data, fname=None):
    """ Plots the data, either using plt.show or saves to a file.

    Args:
        data: The data produced by sim_data
        fname: If None, plot to screen, else save figure as fname.

    Returns: Nothing

    """
    x = data['ss']
    xx = np.linspace(data['ss'][0], data['ss'][-1]*1.05, 1000)

    # Analytic curve
    y = 2.0 / (x ** 2)
    yy = 2.0 / (xx ** 2)

    # Get the data to plot
    y1 = np.array(data['FI_deft_median'])
    y1_rel_err = np.array(data['err_deft_median'])
    y2 = np.array(data['FI_kde_median'])
    y2_rel_err = np.array(data['err_kde_median'])
    y1_err = [np.array(y1-data['FI_deft_5']), np.array(data['FI_deft_95'])-y1]
    y2_err = [np.array(y2-data['FI_kde_5']), np.array(data['FI_kde_95'])-y2]
    y1_err_spread = [np.array(y1_rel_err-data['err_deft_5']), np.array(data['err_deft_95'])-y1_rel_err]
    y2_err_spread = [np.array(y2_rel_err-data['err_kde_5']), np.array(data['err_kde_95'])-y2_rel_err]

    # Some plotting settings
    plt.style.use("publication")
    fig = plt.figure()
    fig.set_size_inches(5, 5)

    # Should we skip the first value because it's FI is too high? 0 means no, 1
    # means skip 1, etc...
    skip_first = 1
    y1_err = [y1_err[0][skip_first:], y1_err[1][skip_first:]]
    y2_err = [y2_err[0][skip_first:], y2_err[1][skip_first:]]
    y1_err_spread = [y1_err_spread[0][skip_first:], y1_err_spread[1][skip_first:]]
    y2_err_spread = [y2_err_spread[0][skip_first:], y2_err_spread[1][skip_first:]]

    # Upper plot (showing FI values)
    ax1 = fig.add_subplot(211)
    ax1.plot(xx, yy, "k", lw=2.0, label="True value")
    lw = 1.5
    deft_color = "#00a442"
    ax1.errorbar(x[skip_first:], y1[skip_first:], y1_err, fmt="o", color=deft_color, lw=lw, label="FI (DEFT)")
    ax1.errorbar(x[skip_first:], y2[skip_first:], y2_err, fmt="x", color="#08519c", lw=lw, label="FI (KDE)")
    ax1.set_xlim(0.1, 1.05)
    ax1.set_ylabel("$g_{\sigma\sigma}$")
    ax1.legend(loc='upper right', prop={"size": 8}, numpoints=1)
    ax1.set_ylim(0,100)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.get_xaxis().set_tick_params(direction='in', top=False)

    # Relative errors plot
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.errorbar(x[skip_first:], y1_rel_err[skip_first:], y1_err_spread, fmt="o", lw=lw, color=deft_color, label="DEFT Relative Error")
    ax2.errorbar(x[skip_first:], y2_rel_err[skip_first:], y2_err_spread, fmt="x", lw=lw, color="#08519c", label="KDE Relative Error")
    ax2.get_xaxis().set_tick_params(top=False)

    ax2.set_xlim(0.1, 1.05)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_xlabel("$\sigma$")
    ax2.set_ylabel(r"$\frac{FI-g_{\sigma\sigma}}{g_{\sigma\sigma}}$")
    ax2.legend(loc='upper right', prop={"size": 8}, numpoints=1)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, dpi=700, bbox_inches="tight")

if __name__ == '__main__':

    start_time = timeit.default_timer()

    # Parameters of the plot
    ss = np.linspace(0.1, 1, 10)
    N = 10000
    rep = 100
    e = 0.05
    zero = np.power(10.0, -10)
    G = 100
    alpha = 3
    seed = 100
    np.random.seed(seed)

    fname = "fig2_data_N_%d_rep_%d_e_%.4f_seed_%d.pklz" % (N, rep, e, seed)
    if os.path.isfile(fname):
        print("Found file!")
        f = gzip.open(fname, "rb")
        data = pickle.load(f)
        f.close()
    else:
        print("Didn't find file, simulating...")
        data = simulate_data(ss, N, rep, e, zero, G, alpha, fname)

    if __debug__:
        print("Obtaining the data took %.2f seconds" % (timeit.default_timer()-start_time))

    plot_data(data)
