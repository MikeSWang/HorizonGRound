#!/usr/bin/env python3

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# export.py: EXPORT POWER SPECTRUM MULTIPOLE SIGNATURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Export and visualise possible power spectrum multipole signatures."""

# =============================================================================
# LIBRARY
# =============================================================================

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

DIR = "multipole_signature/collated/"
PREFIX = "multipole_signature"
TAG = "(nbar=0.001,z=0.,side=1000.,nmesh=[cp256],niter=940)-evol"
TAG_ADD = "(nbar=0.001,z=0.,side=1000.,nmesh=[cp256],niter=940)-stat"

SIGNATURE = 'quant'  # 'model', 'likes', 'quant'

COLLATE = False
SAVE = False

LOAD = True
LOAD_ADD = True

AGGREGATE = True

EXPORT = True
SAVEFIG = False

# Collate and/or save data.
if COLLATE:
    results, count, _ = collate(f"{PATHOUT}{DIR}{PREFIX}-*.npy", 'npy')
    if SAVE: np.save(f"{PATHOUT}{DIR}{PREFIX}-{TAG}.npy", results)
    if AGGREGATE: data = aggregate(results)

# Load data.
if LOAD and (TAG is not None):
    results = np.load(f"{PATHOUT}{DIR}{PREFIX}-{TAG}.npy").item()
    if AGGREGATE: data = aggregate(results)

if LOAD_ADD and (TAG_ADD is not None):
    results_add = np.load(f"{PATHOUT}{DIR}{PREFIX}-{TAG_ADD}.npy").item()
    if AGGREGATE: data_add = aggregate(results_add)

# Remove bad data.
reduced = slice(1, None)
for key, val in data.items():
    data[key] = val[reduced]
for key, val in data_add.items():
    data_add[key] = val[reduced]

# Calculate Kaiser models.
Plin = cosmo.LinearPower(cosmo.Planck15, redshift=0., transfer='CLASS')
growth_rate = cosmo.background.MatterDominated(0.307).f1(1)
bias = 2.

Pk = bias**2 * Plin(data['k'])
beta = growth_rate / bias
model = {
    'P0': (1 + 2/3 * beta + 1/5 * beta**2) * Pk,
    'P2': (4/3 * beta + 4/7 * beta**2) * Pk,
    'P4': (8/35 * beta**2) * Pk,
    }

# Export data.
if EXPORT:
    np.seterr(divide='ignore', invalid='ignore')
    plt.style.use(hgrstyle)
    plt.close('all')
    plt.figure('Multipoles signature')
    sns.set(style='ticks', font='serif')

    ells = [0, 2,]
    lines = {}

    if SIGNATURE == 'model':  # model comparison
        for ell in ells:
            lines[ell] = plt.loglog(
                data['k'], data[f'P{ell}'] / model[f'P{ell}'],
                label=r'$\ell = {{{}}}$'.format(ell)
                )
            plt.fill_between(
                data['k'],
                (data[f'P{ell}'] - data[f'dP{ell}']) / model[f'P{ell}'],
                (data[f'P{ell}'] + data[f'dP{ell}']) / model[f'P{ell}'],
                color=lines[ell][0].get_color(), alpha=1/4
                )
        if LOAD_ADD:
            for ell in ells:
                plt.loglog(
                    data_add['k'], data_add[f'P{ell}'] / model[f'P{ell}'],
                    color=lines[ell][0].get_color(), linestyle='-.'
                    )
                plt.fill_between(
                    data_add['k'],
                    (data_add[f'P{ell}']
                        - data_add[f'dP{ell}']) / model[f'P{ell}'],
                    (data_add[f'P{ell}']
                        + data_add[f'dP{ell}']) / model[f'P{ell}'],
                    color=lines[ell][0].get_color(), alpha=1/4
                    )
        plt.axhline(y=1, c='gray', ls=':')
        plt.ylim(bottom=0.2, top=20)
    elif SIGNATURE == 'likes' and LOAD_ADD:  # like-for-like comparison
        for ell in ells:
            ratio = data[f'P{ell}'] / data_add[f'P{ell}']
            ratio_lower = (data[f'P{ell}'] - data[f'dP{ell}']) \
                / (data_add[f'P{ell}'] + data_add[f'dP{ell}'])
            ratio_upper = (data[f'P{ell}'] + data[f'dP{ell}']) \
                / (data_add[f'P{ell}'] - data_add[f'dP{ell}'])
            lines[ell] = plt.loglog(
                data['k'], ratio, label=r'$\ell = {{{}}}$'.format(ell)
                )
            plt.fill_between(
                data['k'], ratio_lower, ratio_upper,
                color=lines[ell][0].get_color(), alpha=1/4
                )
        plt.axhline(y=1, c='gray', ls=':')
        plt.ylim(bottom=0.2, top=20)
    elif SIGNATURE == 'quant' and LOAD_ADD:  # quantitative comparison
        lines = {}
        for ell in ells:
            deviation = data[f'P{ell}'] - data_add[f'P{ell}']
            lines[ell] = plt.loglog(
                data['k'], deviation / Pk,
                label=r'$\ell = {{{}}}$'.format(ell)
                )

    # Annotation.
    plt.xlim(left=min([d['k'][0] for d in [data, data_add]]),
             right=max([d['k'][-1] for d in [data, data_add]]),
             )
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(
        r'$P_{\ell,\mathrm{evol}}(k) / P_{\ell,\mathrm{stat}}(k)$ '
        r'[$(\textrm{Mpc}/h)^3$]'
        )

    if SIGNATURE == 'model' and LOAD_ADD:
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
