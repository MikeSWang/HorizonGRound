#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# multipole_signature.py: EVOLUTION SIGNATURE IN MULTIPOLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Seeek signatures of evolving background number density in power spectrum
multipoles.
"""

# =============================================================================
# LIBRARY
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
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
    y : float, array_like
        Function value.

    """
    y = 1 + slope * (x - xmin) / (xmax - xmin)

    return y


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
    selection : bool or float, array_like
        Selection value.

    """
    x = np.atleast_1d(x)
    if x.ndim != 1:
        x = np.squeeze(x)

    if mode.lower().startswith('d'):
        selection = density_func(x, *args, **kargs)
    elif mode.lower().startswith('r'):
        selection = (np.random.rand(len(x)) < density_func(x, *args, **kargs))

    return selection


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
KMAX = 0.1

Plin = cosmo.LinearPower(cosmo.Planck15, redshift=Z, transfer='CLASS')
growth_rate = cosmo.background.MatterDominated(0.307).f1(1)


# PROCESSING
# -----------------------------------------------------------------------------

stat = {'k': [], 'Nk': [], 'P0': [], 'P2': [], 'P4': [],}
evol = {'k': [], 'Nk': [], 'P0': [], 'P2': [], 'P4': [],}

for run in range(NITER):
    # Generate evolution catalogue.
    clog_evol = LogNormalCatalog(Plin, NBAR, BOXSIDE, NMESHC)

    clog_evol['Weight'] = select_to_density(
        clog_evol['Position'][:, -1], select_to_density, 'definite', 0, BOXSIDE
        )
    clog_evol['Position'] += clog_evol['VelocityOffset'] * [0, 0, 1]

    mesh_evol = clog_evol.to_mesh(
        Nmesh=NMESHF, resampler='tsc', compensated=True, interlaced=True
        )

    # Generate static catalogue.
    clog_stat = LogNormalCatalog(
        Plin, NBAR, BOXSIDE, NMESHC, seed=clog_evol.attrs['seed']
        )

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

    stat['k'].append(poles_evol['k'])
    stat['Nk'].append(poles_evol['modes'])
    stat['P0'].append(
        poles_evol['power_0'].real - poles_evol.attrs['shotnoise']
        )
    stat['P2'].append(poles_evol['power_2'].real)
    stat['P4'].append(poles_evol['power_4'].real)


# FINALISATION
# -----------------------------------------------------------------------------

# Export data.
np.save(f"{PATHOUT}{DIR}{PREFIX}-{TAG}-evol.npy", evol)
np.save(f"{PATHOUT}{DIR}{PREFIX}-{TAG}-stat.npy", stat)

# Visualise data
plt.style.use(hgrstyle)
plt.close('all')
plt.figure('Multipoles signature')

Pk = Plin(np.array(evol['k'][0] + stat['k'][0])/2)
P0 = (1 + 2/3 * growth_rate + 1/5 * growth_rate**2) * Pk
P2 = (4/3 * growth_rate + 4/7 * growth_rate**2) * Pk
P4 = (8/35 * growth_rate**2) * Pk

with np.errstate(divide='ignore'):
    p0_evol = plt.loglog(
        evol['k'], evol['P0']/P0, label=r'$\ell = 0$'
        )
    p2_evol = plt.loglog(
        evol['k'], evol['P2']/P2, label=r'$\ell = 2$'
        )
    p4_evol = plt.loglog(
        evol['k'], evol['P4']/P4, label=r'$\ell = 4$'
        )

    p0_stat = plt.loglog(
        stat['k'], stat['P0']/P0, color=p0_evol[0].get_color(), ls='--'
        )
    p2_stat = plt.loglog(
        stat['k'], stat['P2']/P2, color=p2_evol[0].get_color(), ls='--'
        )
    p4_stat = plt.loglog(
        stat['k'], stat['P4']/P4, color=p4_evol[0].get_color(), ls='--'
        )

line_styles = ['-', '--']
line_labels = ['evolution', 'static']
lines = [Line2D([0], [0], color='k', linestyle=ls) for ls in line_styles]
handles, labels = plt.gca().get_legend_handles_labels()
handles.extend(lines)
labels.extend(line_labels)

plt.legend(handles, labels)
plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
plt.ylabel(r'$\hat{P}_\ell(k)/P_\ell(k)$ [$(\textrm{Mpc}/h)^3$]')
plt.savefig(f"{PATHOUT}{DIR}{PREFIX}-{TAG}.pdf")
