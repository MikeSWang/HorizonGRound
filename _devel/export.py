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
from nbodykit.lab import cosmology as cosmo

from horizonground.toolkit import collate


# =============================================================================
# DEFINITION
# =============================================================================

def aggregate(result):

    return {
        #'Nk': np.sum(result['Nk'], axis=0)/2,
        'k': np.average(result['k'], axis=0),
        'P0': np.average(result['P0'], axis=0),
        'P2': np.average(result['P2'], axis=0),
        'P4': np.average(result['P4'], axis=0),
        'dk': np.std(result['k'], axis=0, ddof=1),
        'dP0': np.std(result['P0'], axis=0, ddof=1),
        'dP2': np.std(result['P2'], axis=0, ddof=1),
        'dP4': np.std(result['P4'], axis=0, ddof=1),
        }


# =============================================================================
# EXECUTION
# =============================================================================

DIR = "multipole_signature/"
PREFIX = "multipole_signature"
TAG = "(nbar=0.001,z=0.,side=500.,nmesh=[cp256],niter=5)-evol"

COLLATE = True
LOAD = False
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
    if SAVE: np.save(f"{PATHOUT}{PREFIX}-{TAG}.npy", results)
    if AGGREGATE: data = aggregate(results)

# Load data
if LOAD and (TAG is not None):
    results = np.load(f"{PATHOUT}{PREFIX}-{TAG}.npy").item()
    if AGGREGATE: data = aggregate(results)

# Export data
if EXPORT:

    plt.style.use(hgrstyle)
    plt.close('all')
    plt.figure('Multipoles signature')

    Pk = Plin(data['k'])
    P0 = (1 + 2/3 * frate + 1/5 * frate**2) * Pk
    P2 = (4/3 * frate + 4/7 * frate**2) * Pk
    P4 = (8/35 * frate**2) * Pk
    with np.errstate(divide='ignore'):
        p0_line = plt.loglog(data['k'], data['P0']/P0, label=r'$\ell = 0$')
        p2_line = plt.loglog(data['k'], data['P2']/P2, label=r'$\ell = 2$')
        p4_line = plt.loglog(data['k'], data['P4']/P4, label=r'$\ell = 4$')

        plt.fill_between(
            data['k'],
            (data['P0'] - data['dP0'])/P0, (data['P0'] + data['dP0'])/P0,
            color=p0_line[0].get_color(), alpha=0.25
            )
        plt.fill_between(
            data['k'],
            (data['P2'] - data['dP2'])/P2, (data['P2'] + data['dP2'])/P2,
            color=p2_line[0].get_color(), alpha=0.25
            )
        plt.fill_between(
            data['k'],
            (data['P4'] - data['dP4'])/P4, (data['P4'] + data['dP4'])/P4,
            color=p4_line[0].get_color(), alpha=0.25
            )

    plt.legend()
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$\hat{P}_\ell(k)/P_\mathrm{lin}(k)$ [$(\textrm{Mpc}/h)^3$]')

    if SAVEFIG: plt.savefig(f"{PATHOUT}{PREFIX}-{TAG}.pdf")
