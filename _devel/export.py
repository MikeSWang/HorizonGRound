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

from horizonground.toolkit import collate


# =============================================================================
# DEFINITION
# =============================================================================

def aggregate(results):

    data = {}

    return data


# =============================================================================
# EXECUTION
# =============================================================================

DIR = "multipole_signature/"
PREFIX = "multipole_signature"
TAG = "(nbar=,z=,side=,nmesh=[cp256],niter=)"

COLLATE = False
LOAD = True

EXPORT = True

SAVE = False
SAVEFIG = False

# Collate and/or save data
if COLLATE:
    results, count, _ = collate(f"{PATHOUT}{DIR}{PREFIX}-*.npy", 'npy')
    if SAVE:
        np.save(f"{PATHOUT}{PREFIX}-{TAG}.npy", results)

# Load data
if LOAD and (TAG is not None):
    results = np.load(f"{PATHOUT}{PREFIX}-{TAG}.npy").item()

# Export data
if EXPORT:

    plt.style.use(hgrstyle)
    plt.close('all')
    plt.figure('Multipoles signature')

    plt.legend()
    plt.xlabel(r'$k$ [$h/\textrm{Mpc}$]')
    plt.ylabel(r'$\hat{P}_\ell(k)$ [$(\textrm{Mpc}/h)^3$]')

    if SAVEFIG: plt.savefig(f"{PATHOUT}{PREFIX}-{TAG}.pdf")
