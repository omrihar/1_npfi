#!/usr/bin/env python2
# encoding: utf-8

'''
npfi.py (v0.1)

Written by:
    Omri Har-Shemesh, Computational Science Lab, University of Amsterdam
        O.HarShemesh@uva.nl

Last updated on 25 September 2015

Description:
    Estimation of Fisher information based on finite differences and
    non-parametric density estimation.

References:
    [1] O. Har-Shemesh, R. Quax, A.G. Hoekstra, P.M.A. Sloot, Non-parametric
        estimation of Fisher information from real data, arxiv:1507.00964[stat.CO]
    [2] Kinney JB (2014) Estimation of probability densities using scale-free
        field theories. Phys Rev E 90:011301(R). arXiv:1312.6661 [physics.data-an].

Functions:
    npfi
    npfi_1d
    get_bbox
    get_pdfs_from_data

Dependencies:
    numpy
    scipy
    timeit
    deft [2]

'''
from __future__ import division, print_function

# Imports
import numpy as np
from numpy.random import normal
from scipy.integrate import quad

from scipy.stats import gaussian_kde

HAS_DEFT = True
try:
    from deft import deft_1d
except ImportError:
    print("Didn't find DEFT package, disabling method='deft' in PDF computation.")
    print("You can obtain DEFT at:")
    print("https://github.com/jbkinney/13_deft")
    HAS_DEFT = False


import timeit

def npfi(pdfs, dt, bounds=(-np.inf, np.inf), logarithmic=True,
                       zero=1e-10, simple=False, N=None, verbose=False):
    """
    Computes an entry in the Fisher information matrix (FIM).

    Args:
        pdfs: A list of 3 or 5 pdf functions at the points where the FIM should
            be estimated.  When three PDFs are supplied, it is assumed that the
            diagonal element of the FIM is required. If 5 pdfs are supplied,
            then the off-diagonal element is computed. The order with which the
            pdfs appear is assumed to follow the following diagrams:

                3 pdfs:       pdfs[1]   pdfs[0]   pdfs[2]
                -------
                                        pdfs[3]
                5 pdfs:       pdfs[1]   pdfs[0]   pdfs[2]
                -------                 pdfs[4]

            Such that pdfs[0] = pdf(x;theta), pdfs[1] = pdf(x;theta-dtheta),
            pdfs[2] = pdf(x;theta+dtheta) and so on.

        dt: Either a float (if diagonal element) or a tuple, with the parameter
            differences dtheta.

        bounds: This is the integration bounds on which the PDFs are defined. If
            DEFT [2] was used to estimate the PDFs, this is the bbox parameter.
            It is assumed that integration between the bounds of each PDF yields
            unity.

        logarithmic: There are two analytically equivalent formulas, which
            however differ numerically that can be used [1]:
            (1)   I = int p(x;theta) [d/dt1 ln p(x;theta)] * [d/dt2 ln p(x;theta)] dx
            (2)   I = int [d/dt1 p(x;theta)] * [d/dt2 p(x;theta)] dx/p(x;theta)

            When logarithmic = True (the default behavior) Eq.(1) is used, when
            logarithmic = False Eq.(2) is used.

        zero: The numerical definition of zero for the algorithm. If the value of
            any of the densities provided in pdfs is below this value for some
            x, the contribution to the integral is assumed to vanish.

        simple: If True and not logarithmic, whenever any of the pdfs
            given as input has a zero value, returns a zero for this point of
            the integration. If False (default), depending on which of the pdfs
            is zero, return the best approximation of the FIM with one of the
            derivatives canceling (see Ref. [17] in [1] for more details).

        N: The number of samples used to compute each of the pdfs. Used in the
            estimation of the epsilon parameter [1].

        verbose: If verbose prints out debug information such as run times.

    Returns:
        FIM: A number (float) which is the result of the computation.
        int_err: The estimated integration error returned from quad.
        epsilon: If N is not None, the estimated epsilon parameter.

    """
    assert len(pdfs) in [3, 5]
    assert isinstance(dt, float) or \
        (len(dt) == 2 and isinstance(dt[0], float) and isinstance(dt[1], float))

    if verbose:
        start = timeit.default_timer()

    if isinstance(dt, float):
        dt2 = dt ** 2
    else:
        dt2 = dt[0] * dt[1]

    diagonal = len(pdfs) == 3
    if logarithmic: # Define FIM using Eq. (1)
        def fim(x, pdfs, dt2):
            vals = np.zeros(len(pdfs))
            for i, p in enumerate(pdfs):
                vals[i] = p(x)
            if np.any(vals <= zero):
                return 0.0
            if diagonal:
                return vals[0] * ((np.log(vals[2]) - np.log(vals[1]))) ** 2 / (4.0 * dt2)
            else:
                return vals[0] * (np.log(vals[2]) - np.log(vals[1])) * (np.log(vals[4]) - np.log(vals[3])) / (4.0 * dt2)
    else: # Define FIM using Eq. (2)
        def fim(x, pdfs, dt2):
            vals = np.zeros(len(pdfs))
            for i, p in enumerate(pdfs):
                vals[i] = p(x)
            # Because this formula has 1/p (unlike the logarithmic one), we can
            # be smart about what to do when some of the pdfs are zero.
            # If simple, just return zero (meaning, in the limit where the pdfs
            # don't overlap, i.e. because dt is too large, the whole FI will be
            # zero). If not simple, return the appropriate value one would
            # obtain without the discrete approximation. See Ref.[17] in [1] for
            # details.
            if simple:
                if np.any(vals <= zero):
                    return 0.0
            else:
                if vals[0] <= zero:
                    return 0.0
                if diagonal:
                    if vals[2] <= zero:
                        return vals[0]
                else:
                    if vals[2] <= zero and vals[4] > zero:
                        return (vals[4]-vals[3]) / (4.0 * dt2)
                    if vals[2] > zero and vals[4] <= zero:
                        return (vals[2] - vals[1]) / (4.0 * dt2)
                    if vals[2] <= zero and vals[4] <= zero:
                        return vals[0]
            # If none of the above is zero, just return the FI from the finite
            # difference formula.

            if diagonal:
                return (vals[2] - vals[1]) ** 2 / (vals[0] * 4.0 * dt2)
            else:
                return ((vals[2] - vals[1]) * (vals[4] - vals[3])) / \
                        (vals[0] * 4.0 * dt2)

    # Compute the integral
    FIM, int_err = quad(fim, bounds[0], bounds[1], args=(pdfs, dt2), limit=200)

    if verbose:
        print("Estimation of the FIM took: %.2f seconds" % (timeit.default_timer()-start))

    if N is not None:
        epsilon = np.sqrt(2.0 / (N * FIM * dt2))
        return FIM, int_err, epsilon
    return FIM, int_err, None

def npfi_1d(pdfs, dt, bounds=(-np.inf, np.inf), logarithmic=True,
                       zero=1e-10, simple=False, N=None, verbose=False):
    """ Computes the Fisher information along a one dimensional line in
        parameter space. The samples are supposed to be ordered along the line
        with separation dt between each to pdfs.

    Args:
        pdfs: An array of PDFs, with at least three PDFs, that are all lying
            in one line in parameter space, separated by an interval of dt.  The
            PDFs should be functions defined on the interval bounds[0] to
            bounds[1] at least.

        dt: A float, the separation between the PDFs provided in pdfs.

        bounds: This is the integration bounds on which the PDFs are defined. If
            DEFT [2] was used to estimate the PDFs, this is the bbox parameter.
            It is assumed that integration between the bounds of each PDF yields
            unity.

        logarithmic: There are two analytically equivalent formulas, which
            however differ numerically that can be used [1]:
            (1)   I = int p(x;theta) [d/dt1 ln p(x;theta)] * [d/dt2 ln p(x;theta)] dx
            (2)   I = int [d/dt1 p(x;theta)] * [d/dt2 p(x;theta)] dx/p(x;theta)

            When logarithmic = True (the default behavior) Eq.(1) is used, when
            logarithmic = False Eq.(2) is used.

        zero: The numerical definition of zero for the algorithm. If the value of
            any of the densities provided in pdfs is below this value for some
            x, the contribution to the integral is assumed to vanish.

        simple: If True and not logarithmic, whenever any of the pdfs
            given as input has a zero value, returns a zero for this point of
            the integration. If False (default), depending on which of the pdfs
            is zero, return the best approximation of the FIM with one of the
            derivatives canceling (see Ref. [17] in [1] for more details).

        N: The number of samples used to compute each of the pdfs. Used in the
            estimation of the epsilon parameter [1].

        verbose: If verbose prints out debug information such as run times.

    Returns:
        FIs: A numpy array of FI values, starting at the position of pdfs[1] and
            with length len(pdfs)-2.

        eps: A numpy array of epsilon values (see [1]) for each of the returned
            FIs.

    """
    assert len(pdfs) >= 3
    assert isinstance(dt, float)

    shape = len(pdfs) - 2

    FIs = np.zeros(shape=shape)
    eps = np.zeros(shape=shape)
    for i in range(shape):
        FIs[i], err, eps[i] = npfi([pdfs[i+1], pdfs[i], pdfs[i+2]], dt, bounds,
                                    logarithmic, zero, simple, N, verbose)
    return FIs, eps


def get_bbox(samples, multi_dim=True, factor=0.5):
    """ A helper function to compute a bounding box for a given set of samples.
        The bounding box that is returned begins at the minimum value of samples
        and ends at the maximum, adding a factor x length buffer to each side of
        the bounding box. In total, if length = max(samples) - min(samples) the
        bounding box is of size length * (1+2*factor)

    Args:
        samples (list/np.array): Samples to compute bbox for

        multi_dim: Is it a one dimensional set of samples or a list of samples?

        factor (float): factor to increase bounding box

    Returns: A tuple with the computed bounding box

    """
    if multi_dim:
        smin, smax = np.min(samples[0]), np.max(samples[0])
        for s in samples:
            smin = np.min([smin, np.min(s)])
            smax = np.max([smax, np.max(s)])
    else:
        smin, smax = np.min(samples), np.max(samples)
    sint = smax - smin
    bbox = (smin - sint*factor, smax + sint*factor)
    return bbox


def get_pdfs_from_data(data, method="deft", G=200, alpha=3, bbox="adjust", factor=0.5,
                       verbose=False):
    """ Performs a non-parametric estimation of the densities in data and returns
        a list compatible with the npfi function defined above.
        If DEFT is used for the estimates, it uses the same bounding box for all
        PDFs. This bounding box should be used when calling npfi.

    Args:
        data: A list of arrays containing the sample data.

        method: Either "deft" or "gaussian_kde" for the non-parametric
            estimation method.

        G: parameter to be passed to DEFT if used [2].

        alpha: parameter to be passed to DEFT if used [2].

        bbox: Either "adjust" or a bounding box (tuple with two values). Used
            for DEFT only.

        factor: If bbox is "adjust", by which factor to adjust.

        verbose: If set to true, print out debug info such as run times.

    Returns:
        pdfs: a list of the estimated pdfs

        bbox: the appropriate bounding box

    """
    assert hasattr(data, '__iter__')
    assert isinstance(method, str) and method in ["deft", "gaussian_kde"]
    if method == "deft" and not HAS_DEFT:
        raise Exception("DEFT has been disabled.")
    assert isinstance(G, (int, long))
    assert isinstance(alpha, (int, long)) and alpha > 0
    assert isinstance(factor, (int, long, float)) and factor > 0
    assert (isinstance(bbox, str) and bbox == "adjust") or \
            (len(bbox) == 2 and bbox[0] < bbox[1])

    # Track time
    if verbose:
        start = timeit.default_timer()

    # Get the bounding box if necessary
    if method is "deft" and bbox is "adjust":
        bbox = get_bbox(data, multi_dim=True, factor=factor)
    if method is "gaussian_kde":
        bbox = (-np.inf, np.inf)

    # Estimate the PDFs
    pdfs = []
    for d in data:
        if method is "gaussian_kde":
            pdfs.append(gaussian_kde(d))
        else:
            pdfs.append(deft_1d(d, G=G, alpha=alpha, bbox=bbox))

    if verbose:
        print("PDF estimation took %.2f with %s" % (timeit.default_timer()-start, method))

    return pdfs, bbox

if __name__ == '__main__':
    # Compute g_ss for the Gaussian distribution
    s = 1.0
    ds = 0.1
    N = 5000
    rep = 30
    analytic_value = 2.0 / s

    FIMs_kde = []
    FIMs_deft = []
    epsilons_kde = []
    epsilons_deft = []
    for i in range(rep):
        Xa = normal(size=N, scale=s)
        Xb = normal(size=N, scale=s-ds)
        Xc = normal(size=N, scale=s+ds)

        pdfs_deft, bbox_deft = get_pdfs_from_data([Xa, Xb, Xc], method="deft")  # DEFT
        pdfs_kde, bbox_kde = get_pdfs_from_data([Xa, Xb, Xc], method="gaussian_kde")
        FIM_deft, int_err_deft, epsilon_deft = npfi(pdfs_deft, ds, N=N, bounds=bbox_deft, logarithmic=False)
        FIM_kde, int_err_kde, epsilon_kde = npfi(pdfs_kde, ds, N=N, bounds=bbox_kde, logarithmic=True)

        FIMs_deft.append(FIM_deft)
        FIMs_kde.append(FIM_kde)
        epsilons_deft.append(FIM_deft)
        epsilons_kde.append(FIM_kde)

    print("#" * 50)
    print("Estimation of the FI after %d repetitions:" % rep)
    print("Analytic value: %.2f" % analytic_value)
    print("FIM from DEFT: %.3f, epsilon=%.3f" % (np.mean(FIMs_deft), np.mean(epsilon_deft)))
    print("FIM from KDE: %.3f, epsilon=%.3f" % (np.mean(FIMs_kde), np.mean(epsilon_kde)))
    rel_deft = (np.mean(FIMs_deft) - analytic_value) / analytic_value
    rel_deft_95 = (np.percentile(FIMs_deft, 95) - analytic_value) / analytic_value - rel_deft
    rel_deft_5 = rel_deft - (np.percentile(FIMs_deft, 5) - analytic_value) / analytic_value
    print("Relative error DEFT: %.5f + %.5f - %.5f" % (rel_deft, rel_deft_95, rel_deft_5))
    rel_kde = (np.mean(FIMs_kde) - analytic_value) / analytic_value
    rel_kde_95 = (np.percentile(FIMs_kde, 95) - analytic_value) / analytic_value - rel_kde
    rel_kde_5 = rel_kde - (np.percentile(FIMs_kde, 5) - analytic_value) / analytic_value
    print("Relative error KDE: %.5f + %.5f - %.5f" % (rel_kde, rel_kde_95, rel_kde_5))
    print("#" * 50)

