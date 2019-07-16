#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# export_signature.py: EXPORT POWER SPECTRUM MULTIPOLE SIGNATURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Export and visualise possible power spectrum multipole signatures."""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from nbodykit.lab import cosmology as cosmo

from runconf import PATHOUT, hgrstyle
from horizonground.toolkit import collate


# =============================================================================
# DEFINITION
# =============================================================================

def aggregate(result):

    dof = np.size(np.atleast_2d(result['P0']), axis=0) - 1
    if dof == 0:
        raise ValueError(
            "Insufficient sample size for using standard deviation. "
            )

    return {
        'Nk': np.sum(result['Nk'], axis=0)/2,
        'k': np.average(result['k'], axis=0),
        'P0': np.average(result['P0'], axis=0),
        'P2': np.average(result['P2'], axis=0),
        'P4': np.average(result['P4'], axis=0),
        'dk': np.std(result['k'], axis=0, ddof=1) / np.sqrt(dof),
        'dP0': np.std(result['P0'], axis=0, ddof=1) / np.sqrt(dof),
        'dP2': np.std(result['P2'], axis=0, ddof=1) / np.sqrt(dof),
        'dP4': np.std(result['P4'], axis=0, ddof=1) / np.sqrt(dof),
        }


# =============================================================================
# EXECUTION
# =============================================================================

SUBDIR = "collated"  # "evol", "stat", "collated"
SIGNATURE = 'quant'  # 'model', 'likes', 'quant'

ALSOLOAD = False
SAVEFIG = False

TAG = "(nbar=0.001,z=0.,side=1000.,nmesh=[cp256],niter=1000)-evol"
TAG_ADD = "(nbar=0.001,z=0.,side=1000.,nmesh=[cp256],niter=1000)-stat"


# I/O tasks {True, False}
# -----------------------------------------------------------------------------

PREFIX = "multipole_signature"
DIR = f"{PREFIX}/{SUBDIR}/"

if SUBDIR == "collated":
    COLLATE = False
    SAVE = False
    LOAD = True
    ALSOLOAD = True  # False
else:
    COLLATE = True
    SAVE = True
    LOAD = False
    ALSOLOAD = False


# Data processing tasks
# -----------------------------------------------------------------------------

if COLLATE:
    results, count, _ = collate(f"{PATHOUT}{DIR}{PREFIX}-*.npy", 'npy')
    data = aggregate(results)
    if SAVE: np.save(f"{PATHOUT}{DIR}{PREFIX}-{TAG}.npy", results)

if LOAD:
    results = np.load(f"{PATHOUT}{DIR}{PREFIX}-{TAG}.npy").item()
    data = aggregate(results)

data_add = None
if ALSOLOAD:
    results_add = np.load(f"{PATHOUT}{DIR}{PREFIX}-{TAG_ADD}.npy").item()
    data_add = aggregate(results_add)

# HACK: Remove bad data.
reduced = slice(1, None)
for key, val in data.items():
    data[key] = val[reduced]
if data_add is not None:
    for key, val in data_add.items():
        data_add[key] = val[reduced]


# Data modelling tasks
# -----------------------------------------------------------------------------

Plin = cosmo.LinearPower(cosmo.Planck15, redshift=0., transfer='CLASS')
growth_rate = cosmo.background.MatterDominated(0.307).f1(1)
bias = 2.
evol = -0.5

Pk = bias**2 * Plin(data['k'])
beta = growth_rate / bias
model = {
    'P0': (1 + 2/3 * beta + 1/5 * beta**2) * Pk,
    'P2': (4/3 * beta + 4/7 * beta**2) * Pk,
    'P4': (8/35 * beta**2) * Pk,
    }
modifying_factor = {
    'P0': evol**2 * beta**2 / 3,
    'P2': 2 * evol**2 * beta**2 / 3,
    'P4': 1,
    }  # !!!: Catalogues not matched to this!


# Data export tasks
# -----------------------------------------------------------------------------

np.seterr(divide='ignore', invalid='ignore')

sns.set(style='ticks', font='serif')
plt.style.use(hgrstyle)

plt.close('all')
plt.figure(f'signature {SIGNATURE}')

ells = [0, 2,]
elllines = {}

if SIGNATURE == 'model':  # model comparison
    for ell in ells:
        elllines[ell] = plt.loglog(
            data['k'], data[f'P{ell}'] / model[f'P{ell}'],
            label=r'$\ell = {{{}}}$'.format(ell)
            )
        plt.fill_between(
            data['k'],
            (data[f'P{ell}'] - data[f'dP{ell}']) / model[f'P{ell}'],
            (data[f'P{ell}'] + data[f'dP{ell}']) / model[f'P{ell}'],
            color=elllines[ell][0].get_color(), alpha=1/4
            )
    if ALSOLOAD:
        for ell in ells:
            plt.loglog(
                data_add['k'], data_add[f'P{ell}'] / model[f'P{ell}'],
                color=elllines[ell][0].get_color(), linestyle='-.'
                )
            plt.fill_between(
                data_add['k'],
                (data_add[f'P{ell}']
                    - data_add[f'dP{ell}']) / model[f'P{ell}'],
                (data_add[f'P{ell}']
                    + data_add[f'dP{ell}']) / model[f'P{ell}'],
                color=elllines[ell][0].get_color(), alpha=1/4
                )
    plt.axhline(y=1, c='gray', ls=':')
    plt.ylim(bottom=0.2, top=20)
    plt.ylabel(
        r'$P_{\ell,\mathrm{evol}}(k) / P_{\ell,\mathrm{stat}}(k)$ '
        r'[$(\textrm{Mpc}/h)^3$]'
        )
elif SIGNATURE == 'likes' and ALSOLOAD:  # like-for-like comparison
    for ell in ells:
        ratio = data[f'P{ell}'] / data_add[f'P{ell}']
        ratio_lower = (data[f'P{ell}'] - data[f'dP{ell}']) \
            / (data_add[f'P{ell}'] + data_add[f'dP{ell}'])
        ratio_upper = (data[f'P{ell}'] + data[f'dP{ell}']) \
            / (data_add[f'P{ell}'] - data_add[f'dP{ell}'])
        elllines[ell] = plt.loglog(
            data['k'], ratio, label=r'$\ell = {{{}}}$'.format(ell)
            )
        plt.fill_between(
            data['k'], ratio_lower, ratio_upper,
            color=elllines[ell][0].get_color(), alpha=1/4
            )
    plt.axhline(y=1, c='gray', ls=':')
    plt.ylim(bottom=0.2, top=20)
    plt.ylabel(
        r'$P_{\ell,\mathrm{evol}}(k) / P_{\ell,\mathrm{stat}}(k)$ '
        r'[$(\textrm{Mpc}/h)^3$]'
        )
elif SIGNATURE == 'quant' and ALSOLOAD:  # quantitative comparison
    coeff = [1/3, 2/3,]
    for ell, num in zip(ells, coeff):
        deviation = data[f'P{ell}'] - data_add[f'P{ell}']
        elllines[ell] = plt.loglog(
            data['k'], deviation/Pk,
            label=r'$\ell = {{{}}}$'.format(ell)
            )
        plt.loglog(
            data['k'], modifying_factor[f'P{ell}']/data['k']**2,
            color=elllines[ell][0].get_color(), ls=':',
            label=r'$\ell = {{{}}}$'.format(ell)
            )
    plt.ylabel(
        r'$\Delta P_{\ell,\mathrm{evol}}(k) / P(k)$ '
        r'[$(\textrm{Mpc}/h)^3$]'
        )

# Annotation.
plt.xlim(left=min([d['k'][0] for d in [data, data_add]]),
         right=max([d['k'][-1] for d in [data, data_add]]),
         )
plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')

if SIGNATURE == 'model' and ALSOLOAD:
    reflinestyles = ['-', '-.']
    reflabels = ['evolution', 'static']
    reflines = [
        Line2D([0], [0], color='k', linestyle=refls)
        for refls in reflinestyles
        ]
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(reflines)
    labels.extend(reflabels)
    plt.legend(handles, labels)
elif SIGNATURE == 'likes' or SIGNATURE == 'quant':
    plt.legend()

if SAVEFIG: plt.savefig(f"{PATHOUT}{PREFIX}-{TAG}.pdf")
