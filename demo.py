#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *****************************************************************************
# demo.py: DEMONSTRATION
#
# Author: MS Wang
# Created: 2019-03
# *****************************************************************************

"""Demonstration script."""

# =============================================================================
# LIBRARY
# =============================================================================

import numpy as np

from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower, LogNormalCatalog, UniformCatalog

from style import mplstyle


# =============================================================================
# DEFINITION
# =============================================================================

def sloped_probability(x, xmin, xmax, slope=-0.5):
    """Linear probability density given a slope, normalised to its maximum.

    Parameters
    ----------
    x : float, array_like
        Random variable value
    xmin, xmax : float
        Random variable domain boundaries
    slope : float, optional
        Slope of the linear density (default is -0.5)

    Returns
    -------
    density : float, array_like
        Probability density
    """

    assert((xmin < x).all() and (x < xmax).all())  # check variables in domain
    assert(-1 < slope < 0)  # assert gentle negative slop

    density = 1 + slope * (x-xmin) / (xmax-xmin)  # compute (normalised) density

    return density


def select_to_prob(x, prob_density, *args, **kargs):
    """Random selection given a probability density function.

    Parameters
    ----------
    x : float, array_like
        Random variable value
    prob_density : function
        Probability density function

    ``*args``, ``**kargs`` are additional arguments to be passed to
    :func:`prob_density`.

    Returns
    -------
    selection : bool, array_like
        Selection value
    """

    assert(np.squeeze(x).ndim == 1)  # check variable input is a flat array

    selection = np.random.rand(len(x)) < prob_density(x, *args, **kargs)

    return selection


# =============================================================================
# EXECUTION
# =============================================================================

# Read catalogue information/set up cosmology
# -----------------------------------------------------------------------------

nbar = 5e-3
nmesh = 512
bias = 2
boxsize = 600

cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift=0., transfer="CLASS")
L = cosmo.comoving_distance(0.2)


# Original catalogue
# -----------------------------------------------------------------------------

original_catalogue = LogNormalCatalog(Plin, nbar, bias=bias,
                                      BoxSize=L, Nmesh=nmesh
                                      )
#original_catalogue = UniformCatalog(nbar, boxsize)

original_mesh = original_catalogue.to_mesh(Nmesh=nmesh, resampler='tsc',
                                           compensated=True, interlaced=True
                                           )


# New catalogue
# -----------------------------------------------------------------------------

new_catalogue = LogNormalCatalog(Plin, nbar, bias=bias,
                                 BoxSize=L, Nmesh=nmesh,
                                 seed=original_catalogue.attrs['seed']
                                 )
#new_catalogue = UniformCatalog(nbar, boxsize,
#                               seed=original_catalogue.attrs['seed']
#                               )

# Choose x-axis as LOS and use specified probabiltiy density
new_catalogue['los'] = new_catalogue['Position'][:,0]
new_catalogue['Selection'] = select_to_prob(new_catalogue['los'],
                                            sloped_probability, 0, L,
                                            slope=-0.1
                                            )

new_mesh = new_catalogue.to_mesh(Nmesh=nmesh, resampler='tsc', compensated=True,
                                 interlaced=True
                                 )

# Compute FFT power
# -----------------------------------------------------------------------------

poles_orig = FFTPower(original_mesh, mode='2d', los=[1,0,0], poles=[0]).poles
poles_new = FFTPower(new_mesh, mode='2d', los=[1,0,0], poles=[0]).poles

k_orig = poles_orig['k']
P0_orig = poles_orig['power_0'] - poles_orig.attrs['shotnoise']
k_new = poles_new['k']
P0_new = poles_new['power_0'] - poles_new.attrs['shotnoise']

plt.style.use(mplstyle)
plt.close('all')
plt.figure('Power multipole comparison')

plt.loglog(k_orig, bias**2*Plin(k_orig), ':', label='input')
plt.loglog(k_orig, P0_orig.real, '-+', label='original')
plt.loglog(k_new, P0_new.real, '-x', label='changed')

plt.legend()
plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
plt.ylabel(r'$\hat{P}_0(k)$ [$(\textrm{Mpc}/h)^3$]')
plt.savefig('./Output/Lognormal_monopole_comparison.pdf')
