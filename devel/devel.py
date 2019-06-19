#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# *****************************************************************************
# devel.py: DEVELOPMENT
#
# Author: MS Wang
# Created: 2019-03
# *****************************************************************************

"""Codes currently under development."""

# =============================================================================
# LIBRARY
# =============================================================================

import sys

import numpy as np

from matplotlib import pyplot as plt
from mpi4py import MPI
from nbodykit.lab import cosmology, FFTPower, LogNormalCatalog

from studio import horizon_style


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

    #assert((xmin < x).all() and (x < xmax).all())  # check variables in domain
    #assert(-1 < slope < 0)  # assert gentle negative slop

    density = 1 + slope * (x-xmin) / (xmax-xmin)  # compute normalised density

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

# INITIALISATION
# -----------------------------------------------------------------------------

# System resources
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Input parameters
try:  # attempt to read from command line
    niter = int(sys.argv[1])
    nbar, nmesh = float(sys.argv[2]), float(sys.argv[3])
except:  # resort to defaults
    niter, nbar, nmesh = 10, 5e-3, 256
    sys.argv.extend([str(niter), str(nbar), str(nmesh)])

# Cosmology
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift=0., transfer="CLASS")
bias = 2
L = cosmo.comoving_distance(0.2)

# Output variables
k0_all, k1_all, P0_all, P1_all = None, None, None, None


# PROCESSING
# -----------------------------------------------------------------------------

comm.Barrier()

k0_list, k1_list, P0_list, P1_list = [], [], [], []
for run in range(int(niter)):
    # Original catalogue
    catalogue0 = LogNormalCatalog(Plin, nbar, bias=bias, BoxSize=L, Nmesh=nmesh
                                  )
    catalogue0['Selection'] = select_to_prob(catalogue0['Position'][:,0],
                                             sloped_probability, 0, L,
                                             slope=-0.2
                                             )
    catalogue0['Position'] = catalogue0['Position'] \
                             + catalogue0['VelocityOffset'] * [1, 0, 0]
    mesh0 = catalogue0.to_mesh(Nmesh=nmesh, resampler='tsc', compensated=True)

    # New catalogue
    catalogue1 = LogNormalCatalog(Plin, nbar, bias=bias, BoxSize=L,
                                  Nmesh=nmesh, seed=catalogue0.attrs['seed']
                                  )
    catalogue1['Position'] = catalogue1['Position'] \
                             + catalogue1['VelocityOffset'] * [1, 0, 0]
    catalogue1['Selection'] = select_to_prob(catalogue1['Position'][:,0],
                                             sloped_probability, 0, L,
                                             slope=-0.2
                                             )
    mesh1 = catalogue1.to_mesh(Nmesh=nmesh, resampler='tsc', compensated=True)

    # Compute FFT power
    Poles0 = FFTPower(mesh0, mode='2d', los=[1,0,0], poles=[0,2]).poles
    Poles1 = FFTPower(mesh1, mode='2d', los=[1,0,0], poles=[0,2]).poles

    monopole0 = Poles0['power_2'].real #- Poles0.attrs['shotnoise']
    monopole1 = Poles1['power_2'].real #- Poles1.attrs['shotnoise']

    # Append reordered results
    k0_list.append(Poles0['k'])
    k1_list.append(Poles1['k'])
    P0_list.append(monopole0)
    P1_list.append(monopole1)


# RESULTS
# -----------------------------------------------------------------------------

k0_all = comm.gather(np.asarray(k0_list), root=0)
k1_all = comm.gather(np.asarray(k1_list), root=0)
P0_all = comm.gather(np.asarray(P0_list), root=0)
P1_all = comm.gather(np.asarray(P1_list), root=0)

if rank == 0:
    # Collate and save
    k0_all = np.concatenate(k0_all)
    k1_all = np.concatenate(k1_all)
    P0_all = np.concatenate(P0_all)
    P1_all = np.concatenate(P1_all)

    result = {'k_o': k0_all, 'P0_o': P0_all,
              'k_n': k1_all, 'P0_n': P1_all
              }
    np.save('./output/result-%s.npy' % sys.argv[1:], result)

    # Summarise and visualise
    data = {'k_o': np.average(result['k_o'], axis=0),
            'dk_o': np.std(result['k_o'], axis=0, ddof=1),
            'P0_o': np.average(result['P0_o'], axis=0),
            'dP0_o': np.std(result['P0_o'], axis=0, ddof=1),
            'dof_o':  np.size(result['P0_o'], axis=0) - 1,
            'k_n': np.average(result['k_n'], axis=0),
            'dk_n': np.std(result['k_n'], axis=0, ddof=1),
            'P0_n': np.average(result['P0_n'], axis=0),
            'dP0_n': np.std(result['P0_n'], axis=0, ddof=1),
            'dof_n':  np.size(result['P0_n'], axis=0) - 1
            }

    plt.style.use(horizon_style)
    plt.close('all')
    plt.figure('Monopole comparison')

    plt.errorbar(data['k_o'], data['P0_o'],
                 xerr=data['dk_o']/np.sqrt(data['dof_o']),
                 yerr=data['dP0_o']/np.sqrt(data['dof_o']),
                 elinewidth=.8, label='original'
                 )
    plt.errorbar(data['k_n'], data['P0_n'],
                 xerr=data['dk_n']/np.sqrt(data['dof_n']),
                 yerr=data['dP0_n']/np.sqrt(data['dof_n']),
                 elinewidth=.8, label='reselected'
                 )
    plt.loglog(data['k_o'], bias**2*Plin(data['k_o']), ':', label='input')

    plt.legend()
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$\hat{P}_2(k)$ [$(\textrm{Mpc}/h)^3$]')
    plt.savefig('./output/Multipole_comparison_lognormal.pdf')
