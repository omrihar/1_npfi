README
======

This repository provides a software implementation in Python of the computation
of the Non-Parametric estimation of Fisher Information (NPFI), as presented in
[1]. The repository contains a simple example of how npfi can be used to
compute the Fisher information of the Gaussian distribution, in addition to the
scripts recreating Figs. 2-4 in [1].

Written by Omri Har-Shemesh, Computational Science Lab, University of Amsterdam

Usage
-----

To compute the Fisher information from samples there are two steps necessary -
computing probability density functions (pdfs) from the samples that are taken
at known parameter values and combining these to compute the Fisher
information. The file `npfi.py` provides three two functions, one for each of
these steps and one to integrate the FI along a line. To use these functions
simply place `npfi.py` in the same directory as your script and import the
necessary functions. For example:

```python
    from npfi import npfi, get_pdfs_from_data
```
The process of extracting the PDFs and computing the Fisher information is
described in the following subsections:

### Estimating the PDFs using `get_pdfs_from_data`

The function `get_pdfs_from_data` currently uses either `gaussian_kde` from the
`scipy` Python package or `deft` (if present) to estimate the PDFs for each of
the data provided. It takes as first argument a `list` of `numpy` arrays, each
array assumed to be a list of samples from the same parameter range and returns
a list of pdfs with the corresponding non-parametric estimate for each of
these. The default estimation method is `deft` and the rest of the parameters
it accepts control the estimation process. See the code for a detailed
explanation of each of the parameters. In order for the deft method to work, 
the file `deft.py` which is available at https://github.com/jbkinney/13_deft
has to be placed in the same directory as `npfi.py` and your script.

### Computing the FI from the PDFs

The function `npfi` accepts either three or five pdfs and computes either the
diagonal FI element or the off-diagonal element respectively of the Fisher
information matrix. See the source code for exact implementation details and
documentation of each of the input parameters.

### Example: computing the $g_\sigma\sigma$ component of the FIM
```python
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

```

### Reproducing the figures

To reproduce the figures, simply run `fig2.py`, `fig3.py`, or `fig4.py` using
`python2`.

Files
-----

npfi.py:
    The main file. It provides two functions `npfi` and `get_pdfs_from_data`
    It is independent from the rest of the files in the repository and should
    be placed in the same directory as your code. If `deft.py` is available [2],
    it will be able to use DEFT for the density estimation.

fig2.py:
    Reproduces Fig. 2 in [1] by simulating the data and computing the FI. Note: this does not use the same seed as the original plot in the publication.

fig3.py:
    Reproduces Fig. 3 in [1] by simulating the data and computing the FI. Note: this does not use the same seed as the original plot in the publication.

fig4.py:
    Reproduces Fig. 4 in [1] by simulating the data and computing the FI. Note: this does not use the same seed as the original plot in the publication.


Code
----
https://github.com/omrihar/1_npfi

References
----------
[1] O. Har-Shemesh, R. Quax, A.G. Hoekstra, P.M.A. Sloot, Non-parametric
        estimation of Fisher information from real data, (2015) arxiv:1507.00964[stat.CO]

