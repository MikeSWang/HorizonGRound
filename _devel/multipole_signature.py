#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# devel.py: DEVELOPMENT SCRIPT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Development codes."""

from runconf import PATHOUT, argv, filename


# =============================================================================
# LIBRARY
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology as cosmo, LogNormalCatalog, FFTPower

from horizonground.studio import horizon_style as hgrstyle
from horizonground.toolkit import float_format as ff


# =============================================================================
# DEFINITION
# =============================================================================

def sloped_probability(x, xmin, xmax, slope=-0.5):
    """Linear probability density given a slope, normalised to its maximum.

    Parameters
    ----------
    x : float, array_like
        Random variable value.
    xmin, xmax : float
        Random variable domain boundaries.
    slope : float, optional
        Slope of the linear density (default is -0.5).

    Returns
    -------
    density : float, array_like
        Probability density.

    """
    density = 1 + slope * (x - xmin) / (xmax - xmin)

    return density


def select_to_prob(x, prob_density, *args, **kargs):
    """Random selection given a probability density function.

    Parameters
    ----------
    x : float, array_like
        Random variable value
    prob_density : function
        Probability density function
    *args, **kargs :
        Additional arguments to be passed to `prob_density`.

    Returns
    -------
    selection : bool, array_like
        Selection value.

    """
    x = np.atleast_1d(x)
    if x.ndim != 1:
        x = np.squeeze(x)

    selection = np.random.rand(len(x)) < prob_density(x, *args, **kargs)

    return selection


# =============================================================================
# EXECUTION
# =============================================================================

# INITIALISATION
# -----------------------------------------------------------------------------

# Runtime variables: read from command line, else resort to defaults.
try:

    NBAR, REDSHIFT, BOXSIDE = float(argv[1]), float(argv[2]), float(argv[3])
    NMESHC, NMESHF, NITER = int(argv[4]), int(argv[5]), int(argv[6])
except:
    NBAR, REDSHIFT, BOXSIDE = 1e-3, 0., 500.
    NMESHC, NMESHF, NITER = 256, 256, 5
    argv.extend([
        str(NBAR), str(REDSHIFT), str(BOXSIDE),
        str(NMESHC), str(NMESHF), str(NITER)
        ])

DIR = f"{filename()}/"
PREFIX = f"{filename()}"

if NMESHC == NMESHF:
    MESH_TAG = f"cp{NMESHC}"
else:
    MESH_TAG = f"c{NMESHC},p{NMESHF}"
TAG = (
   f"(nbar={ff(NBAR, 'sci')},side={ff(BOXSIDE, 'intdot')},"
   f"nmesh=[{MESH_TAG}],niter={NITER})"
   )
if argv[7:]: TAG += f"-{argv[7:]}"

# Runtime constants.
Plin = cosmo.LinearPower(cosmo.Planck15, redshift=REDSHIFT, transfer='CLASS')


# PROCESSING
# -----------------------------------------------------------------------------

output = {
    'k_evol': [], 'P0_evol': [], 'P2_evol': [], 'P4_evol': [],
    # 'k_stat': [], 'P0_stat': [], 'P2_stat': [], 'P4_stat': [],
    }

for run in range(NITER):
    # Evolution catalogue.
    clog_evol = LogNormalCatalog(Plin, NBAR, BOXSIDE, NMESHC)

    clog_evol['Selection'] = select_to_prob(
        clog_evol['Position'][:,-1], sloped_probability, 0, BOXSIDE
        )
    clog_evol['Position'] = clog_evol['Position'] \
        + clog_evol['VelocityOffset'] * [0, 0, 1]

    mesh_evol = clog_evol.to_mesh(
        Nmesh=NMESHF, resampler='tsc', compensated=True, interlaced=True
        )

    # Static catalogue.
    """clog_stat = LogNormalCatalog(
        Plin, NBAR, BOXSIDE, NMESHC, seed=clog_evol.attrs['seed']
        )
    """
    # Compute multipoles.
    poles_evol = FFTPower(
        clog_evol, mode='2d', los=[0, 0, 1], poles=[0, 2, 4]
        ).poles

    # Append reordered results
    output['k_evol'].append(poles_evol['k'])
    output['P0_evol'].append(
        poles_evol['power_0'] - poles_evol.attrs['shotnoise']
        )
    output['P2_evol'].append(poles_evol['power_2'])
    output['P4_evol'].append(poles_evol['power_4'])


# FINALISATION
# -----------------------------------------------------------------------------

# Export data.
np.save(f"{PATHOUT}{DIR}{PREFIX}-{TAG}.npy", output)
