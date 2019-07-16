#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# multipole_signature.py: EVOLUTION SIGNATURE IN MULTIPOLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Seek signatures of evolving background number density in power spectrum
multipoles.
"""

# =============================================================================
# LIBRARY
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology as cosmo, LogNormalCatalog, FFTPower

from runconf import PATHOUT, argv, ff, hgrstyle, filename


# =============================================================================
# DEFINITION
# =============================================================================

def linear_slope(x, xmin, xmax, slope=-0.5):
    """Linear function :math:`y(x)` of given slope normalised to :math:`[0,1]`.

    Parameters
    ----------
    x : float, array_like
        Variable value.
    xmin, xmax : float
        Domain boundaries.
    slope : float, optional
        Normalised slope of the linear density (default is -0.5).

    Returns
    -------
    float, array_like
        Function value.

    """
    return 1 + slope * (x - xmin) / (xmax - xmin)


def select_to_density(x, density_func, mode, *args, **kargs):
    """Selection given a density function, either definite or randomly decided
    by rejection sampling.

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
    bool or float, array_like
        Selection value.

    """
    x = np.atleast_1d(x)
    if x.ndim != 1:
        x = np.squeeze(x)

    if mode.lower().startswith('d'):
        return density_func(x, *args, **kargs)
    elif mode.lower().startswith('r'):
        return (np.random.rand(len(x)) < density_func(x, *args, **kargs))


# =============================================================================
# EXECUTION
# =============================================================================

# INITIALISATION
# -----------------------------------------------------------------------------

# Runtime variables: read from command line, else resort to defaults.
try:

    NBAR, Z, BOXSIDE = float(argv[1]), float(argv[2]), float(argv[3])
    NMESHC, NMESHF, NITER = int(argv[4]), int(argv[5]), int(argv[6])
except:
    NBAR, Z, BOXSIDE = 1e-3, 0., 500.
    NMESHC, NMESHF, NITER = 256, 256, 5
    argv.extend([
        str(NBAR), str(Z), str(BOXSIDE),
        str(NMESHC), str(NMESHF), str(NITER)
        ])

# Runtime identifiers.
DIR = f"{filename()}/"
PREFIX = f"{filename()}"

if NMESHC == NMESHF: MESH_TAG = f"cp{NMESHC}"
else: MESH_TAG = f"c{NMESHC},p{NMESHF}"
TAG = (
   f"(nbar={ff(NBAR, 'sci')},z={ff(Z, 'decdot')},side={ff(BOXSIDE, 'intdot')},"
   f"nmesh=[{MESH_TAG}],niter={NITER})"
   )
if argv[7:]: TAG += f"-{argv[7:]}"

# Runtime constants.
KMAX = 0.5

Plin = cosmo.LinearPower(cosmo.Planck15, redshift=Z, transfer='CLASS')


# PROCESSING
# -----------------------------------------------------------------------------

print(TAG)

stat = {'k': [], 'Nk': [], 'P0': [], 'P2': [], 'P4': [],}
evol = {'k': [], 'Nk': [], 'P0': [], 'P2': [], 'P4': [],}

for run in range(NITER):
    # Generate evolution catalogue.
    clog_evol = LogNormalCatalog(Plin, NBAR, BOXSIDE, NMESHC)

    clog_evol['Weight'] = select_to_density(
        clog_evol['Position'][:, -1], linear_slope, 'definite', 0, BOXSIDE
        )
    clog_evol['Position'] += clog_evol['VelocityOffset'] * [0, 0, 1]

    mesh_evol = clog_evol.to_mesh(
        Nmesh=NMESHF, resampler='tsc', compensated=True, interlaced=True
        )

    # Generate static catalogue.
    clog_stat = LogNormalCatalog(
        Plin, NBAR, BOXSIDE, NMESHC, seed=clog_evol.attrs['seed']
        )
    clog_stat['Position'] += clog_stat['VelocityOffset'] * [0, 0, 1]

    mesh_stat = clog_stat.to_mesh(
        Nmesh=NMESHF, resampler='tsc', compensated=True, interlaced=True
        )

    # Compute multipoles.
    poles_evol = FFTPower(
        mesh_evol, mode='2d', los=[0, 0, 1], poles=[0, 2, 4], kmax=KMAX
        ).poles
    poles_stat = FFTPower(
        mesh_stat, mode='2d', los=[0, 0, 1], poles=[0, 2, 4], kmax=KMAX
        ).poles

    # Append reordered results
    evol['k'].append(poles_evol['k'])
    evol['Nk'].append(poles_evol['modes'])
    evol['P0'].append(
        poles_evol['power_0'].real - poles_evol.attrs['shotnoise']
        )
    evol['P2'].append(poles_evol['power_2'].real)
    evol['P4'].append(poles_evol['power_4'].real)

    stat['k'].append(poles_stat['k'])
    stat['Nk'].append(poles_stat['modes'])
    stat['P0'].append(
        poles_stat['power_0'].real - poles_stat.attrs['shotnoise']
        )
    stat['P2'].append(poles_stat['power_2'].real)
    stat['P4'].append(poles_stat['power_4'].real)


# FINALISATION
# -----------------------------------------------------------------------------

# Export data.
np.save(f"{PATHOUT}{DIR}{PREFIX}-{TAG}-evol.npy", evol)
np.save(f"{PATHOUT}{DIR}{PREFIX}-{TAG}-stat.npy", stat)

# Visualise data.
np.seterr(divide='ignore')
plt.style.use(hgrstyle)
plt.close('all')
plt.figure('Multipoles signature')

k = (np.mean(evol['k'], axis=0) + np.mean(stat['k'], axis=0)) / 2
ells = [0, 2, 4]

for ell in ells:
    ratio = np.mean(evol[f'P{ell}'], axis=0) / np.mean(stat[f'P{ell}'], axis=0)

    plt.loglog(k, ratio, label=r'$\ell = {{{}}}$'.format(ell))

plt.axhline(y=1, c='gray', ls=':')
plt.xlim(right=KMAX)
plt.ylim(bottom=0.2, top=20)
plt.legend()
plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
plt.ylabel(
    r'$P_{\ell,\mathrm{evol}}(k) / P_{\ell,\mathrm{stat}}(k)$ '
    r'[$(\textrm{Mpc}/h)^3$]'
    )
plt.savefig(f"{PATHOUT}{DIR}{PREFIX}-{TAG}.pdf")
