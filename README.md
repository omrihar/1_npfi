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
information. The file `npfi.py` provides two functions, one for each of these
steps. To use these functions simply place `npfi.py` in the same directory as
your script and import the necessary functions. For example:

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
explanation of each of the parameters.

### Computing the FI from the PDFs

The function `npfi` accepts either three or five pdfs and computes either the
diagonal FI element or the off-diagonal element respectively of the Fisher
information matrix. See the source code for exact implementation details and
documentation of each of the input parameters.

Files
-----

npfi.py:
    The main file. It provides two functions `npfi` and `get_pdfs_from_data`
    It is independent from the rest of the files in the repository and should
    be placed in the same directory as your code. If `deft.py` is available [2],
    it will be able to use DEFT for the density estimation.

fig2.py:
    Reproduces Fig. 2 in [1] by simulating the data and computing the FI.

fig3.py:
    Reproduces Fig. 3 in [1] by simulating the data and computing the FI.

fig4.py:
    Reproduces Fig. 4 in [1] by simulating the data and computing the FI.


Code
----
http://github.com/omrihar/1_npfi

References
----------
[1] O. Har-Shemesh, R. Quax, A.G. Hoekstra, P.M.A. Sloot, Non-parametric
        estimation of Fisher information from real data, (2015) arxiv:1507.00964[stat.CO]

