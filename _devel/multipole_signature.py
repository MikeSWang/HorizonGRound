#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# devel.py: DEVELOPMENT SCRIPT  # TODO: Future name change.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Development codes."""  # TODO: Future docstring change.

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

def linear_slope(x, xmin, xmax, slope=-0.5):
    """Linear function :math:`y(x)` of given slope normalised to :math:`[0,1]`.

    Parameters
    ----------
    x : float, array_like
        Independent variable value.
    xmin, xmax : float
        Domain boundaries.
    slope : float, optional
        Slope of the linear density (default is -0.5).

    Returns
    -------
    y : float, array_like
        Function value.

    """
    y = 1 + slope * (x - xmin) / (xmax - xmin)

    return y


def select_to_density(x, density_func, mode, *args, **kargs):
    """Selection given a density function, either definite or randomly sampled
    by rejection.

    Parameters
    ----------
    x : float, array_like
        Variable value.
    density_func : callable
        Density function.
    mode : {'definite', 'random'}
        Definite selection ``'definite'`` or random sampling ``'random'``.
    *args, **kargs
        Positional and keyword parameters to be passed to `density_func`.

    Returns
    -------
    selection : bool or float, array_like
        Selection value.

    """
    x = np.atleast_1d(x)
    if x.ndim != 1:
        x = np.squeeze(x)

    if mode.lower()[:3] == 'def':
        selection = density_func(x, *args, **kargs)
    elif mode.lower()[:3] == 'ran':
        selection = (np.random.rand(len(x)) < density_func(x, *args, **kargs))

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
frate = cosmo.background.MatterDominated(0.307).f1(1)


# PROCESSING
# -----------------------------------------------------------------------------

evol = {'k': [], 'Nk': [], 'P0': [], 'P2': [], 'P4': [],}

for run in range(NITER):
    # Evolution catalogue.
    clog_evol = LogNormalCatalog(Plin, NBAR, BOXSIDE, NMESHC)

    clog_evol['Weight'] = select_to_density(
        clog_evol['Position'][:, -1], linear_slope, 'definite', 0, BOXSIDE
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
        mesh_evol, mode='2d', los=[0, 0, 1], poles=[0, 2, 4]
        ).poles

    # Append reordered results
    evol['k'].append(poles_evol['k'])
    evol['Nk'].append(poles_evol['modes'])
    evol['P0'].append(
        poles_evol['power_0'].real - poles_evol.attrs['shotnoise']
        )
    evol['P2'].append(poles_evol['power_2'].real)
    evol['P4'].append(poles_evol['power_4'].real)


# FINALISATION
# -----------------------------------------------------------------------------

# Export data.
np.save(f"{PATHOUT}{DIR}{PREFIX}-{TAG}-evol.npy", evol)

# Visualise data
plt.style.use(hgrstyle)
plt.close('all')
plt.figure('Multipoles signature')

Pk = Plin(evol['k'])
P0 = (1 + 2/3 * frate + 1/5 * frate**2) * Pk
P2 = (4/3 * frate + 4/7 * frate**2) * Pk
P4 = (8/35 * frate**2) * Pk

with np.errstate(divide='ignore'):
    p0_line = plt.loglog(evol['k'], evol['P0']/P0, label=r'$\ell = 0$')
    p2_line = plt.loglog(evol['k'], evol['P2']/P2, label=r'$\ell = 2$')
    p4_line = plt.loglog(evol['k'], evol['P4']/P4, label=r'$\ell = 4$')

plt.legend()
plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
plt.ylabel(r'$\hat{P}_\ell(k)/P_\ell(k)$ [$(\textrm{Mpc}/h)^3$]')
plt.savefig(f"{PATHOUT}{DIR}{PREFIX}-{TAG}-evol.pdf")
