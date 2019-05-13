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
from nbodykit.lab import FFTPower, UniformCatalog
from nbodykit.source.catalog import CSVCatalog

from style import mplstyle


# =============================================================================
# DEFINITION
# =============================================================================

def sloped_probability(x, xmin, xmax, slope=-0.5):
    """Linear probability density given a slope.

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

    # Compute density
    density = 1 + slope * (x-xmin) / (xmax-xmin)

    # Normalise probability
    normalisation = (1+(1+slope)) * (xmax-xmin) / 2
    density /= normalisation

    return density


def select_to_prob(x, prob_density, *args):
    """Random selection given a probability density function.

    Parameters
    ----------
    x : float, array_like
        Random variable value
    prob_density : function
        Probability density function

    ``*args`` are additional arguments to be passed to :func:`prob_density`.

    Returns
    -------
    decision : bool, array_like
        Selection value
    """

    assert(np.squeeze(x).ndim == 1)  # check variable input is a flat array

    decision = np.random.rand(len(x)) < prob_density(x, *args)

    return decision


# =============================================================================
# EXECUTION
# =============================================================================

# Read catalogue from file
file = './Data/BigMDPL_RockstarHalo_z1.0.txt'
nbar = 1.5e-3
boxsize = 2500
nmesh = 256

# Original catalogue
clog_orig = CSVCatalog(file, ['mass', 'pid', 'x', 'y', 'z', 'vx', 'vy', 'vz'])
clog_orig['Position'] = np.array([clog_orig['x'],
                                  clog_orig['y'],
                                  clog_orig['z']
                                  ]).T

#clog_orig = UniformCatalog(nbar, boxsize)
#clog_orig['x'] = clog_orig.compute(clog_orig['Position'])[:,0]

mesh_orig = clog_orig.to_mesh(BoxSize=boxsize, Nmesh=nmesh, resampler='tsc',
                              compensated=True, interlaced=True
                              )

# Reselect catalogue
clog_new = clog_orig.copy()
clog_new['Selection'] = select_to_prob(clog_orig.compute(clog_orig['x']),
                                       sloped_probability, 0, boxsize, -0.05
                                       )  # use specified probabiltiy density

mesh_new = clog_new.to_mesh(BoxSize=boxsize, Nmesh=nmesh, resampler='tsc',
                            compensated=True, interlaced=True
                            )

# Compute FFT power
power1d_orig = FFTPower(mesh_orig, mode='1d').power
power1d_new = FFTPower(mesh_new, mode='1d').power

k_orig = power1d_orig['k']
Pk_orig = power1d_orig['power'] - power1d_orig.attrs['shotnoise']
k_new = power1d_new['k']
Pk_new = power1d_new['power'] - power1d_new.attrs['shotnoise']

plt.style.use(mplstyle)
plt.close('all')
plt.figure('Power comparison')
plt.loglog(k_orig, Pk_orig.real, '-k', marker='+', label='original')
plt.loglog(k_new, Pk_new.real, '--r', marker='x', label='shuffled')
plt.legend()
plt.xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
plt.ylabel(r'$P(k, z=1.0)$ [$(\mathrm{Mpc}/h)^3$]')
plt.savefig('./Output/power1d_comparison.pdf')
