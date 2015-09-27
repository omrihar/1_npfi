#!/usr/bin/env python2
# encoding: utf-8
'''
fig3.py

Written by:
    Omri Har-Shemesh, Computational Science Lab, University of Amsterdam
        O.HarShemesh@uva.nl

Last updated on 23 September 2015

Description:
    Figure 3 in Ref.[1]

References:
    [1] O. Har-Shemesh, R. Quax, A.G. Hoekstra, P.M.A. Sloot, Non-parametric
        estimation of Fisher information from real data, arxiv:1507.00964[stat.CO]

Functions:
    simulate_data
    plot_date

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

def simulate_data(ss, es, N, rep, zero, G, alpha, fname):
    """ Simulates the data for the plot

    Args:
        ss: An array of sigma values to estimate the FI at.
        es: An array of epsilon values to estimate the FI at.
        N: Number of data points for each PDF.
        rep: Number of repetitions of the whole simulation.
        zero: What should npfi consider as zero
        G: G for DEFT
        alpha: alpha for DEFT
        fname: Name of the file where the simulation data will be stored.

    Returns:
        results: A dictionary with all simulated data, which was also stored to
            the file.
    """
    # Results of the simulation will be stored here
    data = {}

    # Go over all sigma values in ss
    for i, s in enumerate(ss):
        true_fi = 2 / s ** 2

        ess = []  # Store the epsilon values actually used
        dss = []  # Store the ds values we used
        FI_values_all = []
        err_values_all = []
        err_median, err_5, err_95 = [], [], []
        for j, e in enumerate(es):
            ds = s / (e * np.sqrt(N))  # Choose ds according to desired epsilon
            # If ds >= s we have a problem of sampling with negative std
            if ds >= s:
                continue
            dss.append(ds)
            ess.append(e)

            # Estimate the FI for rep repetitions
            FI_values = []
            for j in range(rep):
                sim_data = [normal(size=N, scale=s),
                        normal(size=N, scale=s-ds),
                        normal(size=N, scale=s+ds)]
                pdfs, bbox = get_pdfs_from_data(sim_data, method="deft", G=G,
                                                alpha=alpha, bbox="adjust")
                FI, a, b = npfi(pdfs, ds, bounds=bbox, logarithmic=False,
                                zero=zero, N=N)
                FI_values.append(FI)

            # More convenient to use as numpy arrays
            FI_values = np.array(FI_values)

            # Compute statistics from the results
            err_values = (FI_values - true_fi) / true_fi
            FI_values_all.append(FI_values)
            err_values_all.append(err_values)
            err_median.append(np.median(err_values))
            err_5.append(np.percentile(err_values, 5))
            err_95.append(np.percentile(err_values, 95))

        data[s] = dict(FI_values_all=FI_values_all,
                       err_values_all=err_values_all,
                       err_median=np.array(err_median),
                       err_5=np.array(err_5),
                       err_95=np.array(err_95),
                       dss=dss,
                       ess=ess)

    results = dict(data=data, N=N, rep=rep, ss=ss)
    f = gzip.open(fname, "wb")
    pickle.dump(results, f)
    f.close()

    return results

def plot_data(sim_data, fname=None):
    """ Plots the data, either using plt.show or saves to a file.

    Args:
        sim_data: The data produced by sim_data
        fname: If None, plot to screen, else save figure as fname.

    Returns: Nothing

    """

    # Setup the plotting parameters
    params = {
        'text.usetex' : True,
        'font.size' : 10,
        'font.family' : 'cmr',
        'text.latex.unicode' : True
    }
    plt.rcParams.update(params)
    plt.style.use("publication")
    colors = {
        0 : ["#08519c", "#6baed6", "#3182bd"],
        1 : ["#006d2c", "#66c2a4", "#2ca25f"],
        2 : ["#b30000", "#fdbb84", "#e34a33"],
        3 : ["#54278f", "#9e9ac8", "#756bb1"],
        4 : ["#252525", "#969696", "#cccccc"]
    }

    dot_styles = "o*vsph"

    fig = plt.figure()
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(111)

    i = 0
    for s, data in sim_data['data'].iteritems():
        true_fi = 2.0 / s**2
        x = data['ess']
        y = data['err_median']
        y_5, y_95 = data['err_5'], data['err_95']

        line, = ax.plot(x, y, dot_styles[i] + "-", lw=1.2, markersize=4, color=colors[i][0], label=r"$\sigma=%.1f$" % s)
        ax.fill_between(x, y_5, y_95, alpha=.5, facecolor=colors[i][1], edgecolor=colors[i][2])

        i += 1

    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel(r"$\frac{\mathrm{FI} - g_{\sigma\sigma}}{g_{\sigma\sigma}}$")
    ax.set_xticks([0.01, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(right=False, top=False)
    ax.grid("off")
    ax.set_ylim(-0.75,4)
    ax.set_xlim(0, 0.8)
    ax.legend(loc="upper right", prop={"size": 8})

    # Add inset to the original figure
    add_inset = True
    if add_inset:
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes, inset_axes
        xticks_down = True
        if xticks_down:
            axins = inset_axes(ax, width=1.2, height=1.2, bbox_to_anchor=(0.5, 0.98), bbox_transform=ax.transAxes)
        else:
            axins = inset_axes(ax, width=1.5, height=1.5, bbox_to_anchor=(0.6, 0.95), bbox_transform=ax.transAxes)

        i = 0
        for s, data in sim_data['data'].iteritems():
            true_fi = 2.0 / (s**2)
            x = data['ess']
            y = data['err_median']
            y_5, y_95 = data['err_5'], data['err_95']
            line, = axins.plot(x, y, dot_styles[i] + "-", lw=1.2, markersize=4, color=colors[i][0], label=r"$\sigma=%.1f$" % s)
            line_err = axins.errorbar(x, y, yerr=[y-y_5, y_95-y], ls=dot_styles[i], lw=1.2, markersize=4, color=colors[i][0], label=r"$\sigma=%.1f$" % s)

        axins.set_xlim(0.015, 0.11)
        axins.set_ylim(-0.2, 0.35)
        axins.set_axis_bgcolor("w")
        if xticks_down:
            axins.set_xticks([0.02, 0.04, 0.06, 0.08, 0.1])
            axins.set_xticklabels(["$0.02$", "$0.04$", "$0.06$", "$0.08$", ""])
            axins.get_xaxis().set_tick_params(direction='in', labelcolor="k", labeltop=False, labelbottom=True, labelsize=8)
            axins.get_yaxis().set_tick_params(labelsize=8)
        else:
            axins.get_xaxis().set_tick_params(direction='in', labelcolor="k", labeltop=True, labelbottom=False)
        axins.get_yaxis().set_tick_params(direction='in', labelcolor="k", labelleft=False, labelright=True)
        axins.set_frame_on(True)
        axins.grid("off")
        plt.setp(axins.spines.values(), color="k", lw=1)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="k", lw=1)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, dpi=700, bbox_inches="tight")

if __name__ == '__main__':

    start_time = timeit.default_timer()

    # Parameters of the plot
    rep = 100
    ss = [0.5, 1.0, 2.0, 5.0, 10.0]
    es = np.array([0.01, 0.013, 0.015, 0.017, 0.019, 0.02, 0.03, 0.05,
                   0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                   0.9, 1.0])
    N = 20000
    G = 200
    alpha = 3
    zero = 1e-10
    seed = 200

    np.random.seed(seed)
    fname = "fig3_data_rep_%d_N_%d_seed_%d.pklz" % (rep, N, seed)
    if os.path.isfile(fname):
        print("Found data file, plotting...")
        f = gzip.open(fname, "rb")
        data = pickle.load(f)
        f.close()
    else:
        print("Simulating data...")
        data = simulate_data(ss, es, N, rep, zero, G, alpha, fname)

    if __debug__:
        print("Obtaining the data took %.2f seconds" % (timeit.default_timer()-start_time))

    plot_data(data)
