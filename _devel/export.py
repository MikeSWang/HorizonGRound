#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# export.py: EXPORT POWER SPECTRUM MULTIPOLES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Export results for power spectrum multipoles."""

from runconf import PATHOUT, hgrstyle


# =============================================================================
# LIBRARY
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from nbodykit.lab import cosmology as cosmo

from horizonground.toolkit import collate


# =============================================================================
# DEFINITION
# =============================================================================

def aggregate(result):

    dof = np.size(result['P0'], axis=0) - 1

    return {
        #'Nk': np.sum(result['Nk'], axis=0)/2,
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

DIR = "multipole_signature/"
PREFIX = "multipole_signature"
TAG = "(nbar=0.001,z=0.,side=500.,nmesh=[cp256],niter=1000)-evol"
TAG_ADD = "(nbar=0.001,z=0.,side=500.,nmesh=[cp256],niter=1000)-stat"

COLLATE = False
LOAD = True
LOAD_ADD = True
AGGREGATE = True

EXPORT = True

SAVE = False
SAVEFIG = False

# Runtime constants.
Plin = cosmo.LinearPower(cosmo.Planck15, redshift=0., transfer='CLASS')
frate = cosmo.background.MatterDominated(0.307).f1(1)

# Collate and/or save data
if COLLATE:
    results, count, _ = collate(f"{PATHOUT}{DIR}{PREFIX}-*.npy", 'npy')
    if SAVE: np.save(f"{PATHOUT}{DIR}{PREFIX}-{TAG}.npy", results)
    if AGGREGATE: data = aggregate(results)

# Load data
if LOAD and (TAG is not None):
    results = np.load(f"{PATHOUT}{DIR}{PREFIX}-{TAG}.npy").item()
    if AGGREGATE: data = aggregate(results)

if LOAD_ADD and (TAG_ADD is not None):
    results_add = np.load(f"{PATHOUT}{DIR}{PREFIX}-{TAG_ADD}.npy").item()
    if AGGREGATE: data_add = aggregate(results_add)

# Export data
if EXPORT:
    # Figure property
    plt.style.use(hgrstyle)
    plt.close('all')
    plt.figure('Multipoles signature')

    ell_colours = ['#000000', '#C40233', '#0087BD',]

    # Prediction
    Pk = Plin(data['k'])
    model = {
        'P0': (1 + 2/3 * frate + 1/5 * frate**2) * Pk,
        'P2': (4/3 * frate + 4/7 * frate**2) * Pk,
        'P4': (8/35 * frate**2) * Pk,
        }

    # Comparison
    with np.errstate(divide='ignore'):
        c = iter(ell_colours)
        for ell in [0, 2,]:
            pell_line = plt.loglog(
                data['k'], data[f'P{ell}']/model[f'P{ell}'],
                color=next(c),
                label=r'$\ell = {{{}}}$'.format(ell)
                )
            plt.fill_between(
                data['k'],
                (data[f'P{ell}'] - data[f'dP{ell}'])/model[f'P{ell}'],
                (data[f'P{ell}'] + data[f'dP{ell}'])/model[f'P{ell}'],
                color=pell_line[0].get_color(), alpha=1/8
                )
    if LOAD_ADD:
        with np.errstate(divide='ignore'):
            c_add = iter(ell_colours)
            for ell in [0, 2,]:
                pell_line = plt.loglog(
                    data_add['k'], data_add[f'P{ell}']/model[f'P{ell}'],
                    color=next(c_add),
                    linestyle='-.'
                    )
                plt.fill_between(
                    data_add['k'],
                    (data_add[f'P{ell}'] - data_add[f'dP{ell}'])/model[f'P{ell}'],
                    (data_add[f'P{ell}'] + data_add[f'dP{ell}'])/model[f'P{ell}'],
                    color=pell_line[0].get_color(), alpha=1/8
                    )

    # Annotation
    plt.axhline(y=1, ls='--', c='gray', alpha=0.5)
    plt.xlim(right=0.1)
    plt.ylim(bottom=0.25, top=250)

    linestyles = ['-', '-.']
    linelabels = ['evolution', 'static']
    lines = [
        Line2D([0], [0], color=ell_colours[0], linestyle=ls)
        for ls in linestyles
        ]
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend(lines)
    labels.extend(linelabels)
    plt.legend(handles, labels)

    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$\hat{P}_\ell(k)/P_\ell(k)$ [$(\textrm{Mpc}/h)^3$]')

    if SAVEFIG: plt.savefig(f"{PATHOUT}{PREFIX}-{TAG}.pdf")
