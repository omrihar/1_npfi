#!/usr/bin/env python2
# encoding: utf-8
'''
fig4.py

Written by:
    Omri Har-Shemesh, Computational Science Lab, University of Amsterdam
        O.HarShemesh@uva.nl

Last updated on 25 September 2015

Description:
    Figure 4 in Ref.[1]

References:
    [1] O. Har-Shemesh, R. Quax, A.G. Hoekstra, P.M.A. Sloot, Non-parametric
        estimation of Fisher information from real data, arxiv:1507.00964[stat.CO]
Functions:
    simulate_data
    plot_data

Dependencies:
    numpy
    matplotlib
    timeit
    cPickle
    os
    gzip
    npfi.py

'''
from __future__ import division, print_function

import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import os
import gzip
import cPickle as pickle
import timeit

from npfi import npfi, get_pdfs_from_data

def simulate_data(s, Ns, dss, rep, zero, G, alpha, fname):
    """ Simulates the data for the plot

    Args:
        s: Sigma in which all computations are done.

        Ns: An array of N values where to compute the FI.

        dss: An array of ds to compute with.

        rep: Number of repetitions of the whole simulation.

        zero: What should npfi consider as zero

        G: G for DEFT

        alpha: alpha for DEFT

        fname: Name of the file where the simulation data will be stored.

    Returns:
        data: A dictionary with all simulated data, which was also stored to
            the file.
    """
    # Results of the simulation will be stored here
    shape = (len(Ns), len(dss))
    FIs = np.zeros(shape=shape) + np.nan  # Computed FIs
    err = np.zeros(shape=shape) + np.nan  # Computed absolute relative error
    true_fi = 2.0 / s ** 2
    FI_values_all = []

    # Go over all Ns and dss per N
    for i, N in enumerate(Ns):
        print("Starting %d from %d" % (i+1, len(Ns)))
        FI_row = []
        for j, ds in enumerate(dss):
            # Estimate the FI for rep repetitions
            FI_values = []
            for k in range(rep):
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
            err_values = np.abs(FI_values - true_fi) / true_fi
            FI_row.append(FI_values)

            # Save results in the appropriate matrix
            FIs[i, j] = np.median(FI_values)
            err[i, j] = np.median(err_values)
        FI_values_all.append(FI_row)

    data = dict(FIs=FIs, err=err,Ns=Ns, dss=dss, rep=rep)

    f = gzip.open(fname, "wb")
    pickle.dump(data, f)
    f.close()

    return data

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

    # For contour line labels
    class nf(float):
        def __repr__(self):
            s = "%.1f" % self.__float__()
            if s[-1] == "0":
                return "%.0f" % self.__float__()
            else:
                return "%.1f" % self.__float__()

    x = data['dss']
    y = data['Ns']
    z = data['err']

    # Plot the heatmap and place label of colorbar above it
    plt.imshow(z, extent=[x[0], x[-1], y[0], y[-1]], origin="lower", aspect='auto', norm=LogNorm())
    cb = plt.colorbar()
    label = cb.set_label(r"$\frac{|g_{\sigma\sigma} - \mathrm{FI}|}{g_{\sigma\sigma}}$", rotation="horizontal", fontsize=14)
    cb.ax.yaxis.set_label_coords(0.5,1.07)

    plt.yticks(y)
    plt.xlabel(r"$\Delta\sigma$")
    plt.ylabel(r"$N$")
    lims = plt.gca().get_ylim()

    # Add a line showing epsilon = 0.1 (divided into two for the label)
    xx = np.linspace(x[0], x[-1], 1000)
    Ny = 1.0 / (0.1*xx)**2
    N1, N2 = 30000, 50000
    l1, = plt.plot(xx[Ny < N1], Ny[Ny < N1], lw=2, ls="--", color="k")
    l2, = plt.plot(xx[Ny > N2], Ny[Ny > N2], lw=2, ls=l1.get_linestyle(), color=l1.get_color())

    # add a label to the line
    pos = (0.05, 40000)
    label_str = r"$\varepsilon = 0.1$"
    label_color = l1.get_color()
    label_color = "k"
    txt_bbox = dict(facecolor="w", edgecolor="none", pad=10.0, alpha=.4)
    txt_bbox = None
    txt = plt.text(pos[0], pos[1], label_str, rotation=-85, color=label_color,
             size=16, ha="center", va="center", bbox=txt_bbox)

    # Add a line at $\Delta\sigma = 0.35$
    plt.axvline(0.35, ls="-.", color="k", lw=2)

    # Add the limits again after adding the epsilon=0.1 line
    plt.ylim(lims)
    plt.xlim(data['dss'][0], data['dss'][-1])

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, dpi=700, bbox_inches="tight")

if __name__ == '__main__':

    start_time = timeit.default_timer()

    # Parameters of the plot
    s = 1.0
    G = 200
    alpha = 3
    zero = 1e-10
    seed = 100

    Ns = [1000, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    dss = np.linspace(0.01, 0.5, 30)

    np.random.seed(seed)
    fname = "fig4_data_rep_%d_N_%d_ds_%d_seed_%d.pklz" % (rep, len(Ns), len(dss), seed)
    if os.path.isfile(fname):
        f = gzip.open(fname, "rb")
        data = pickle.load(f)
        f.close()
    else:
        data = simulate_data(s, Ns, dss, rep, zero, G, alpha, fname)

    if __debug__:
        print("Obtaining the data took %.2f seconds" % (timeit.default_timer()-start_time))

    plot_data(data)
