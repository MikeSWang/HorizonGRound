#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================================
# Demo.py: DEMONSTRATION
#
# Author: Mike Shengbo Wang
# Created: 2019-03-12
# ============================================================================


# ================================= LIBRARY ==================================


import numpy as np
import matplotlib.pyplot as plt


from nbodykit.source.catalog import CSVCatalog
from nbodykit.lab import *


import style


# ================================ Definition ================================


def sloped_probability(x, xmin, xmax, slope=-0.5):

    assert(xmin < x < xmax)
    assert(-1 < slope < 0)

    density = 1 + slope * (x - xmin) / (xmax - xmin)
    normalisation = (1 + 1 + slope) * (xmax - xmin) / 2
    density /= normalisation

    return density


def select_decide(x, xmin, xmax):

    p = sloped_probability(x, xmin, xmax)
    if np.random.rand(1) < p:
        decision = True
    else:
        decision = False

    return decision


# ================================ Execution =================================

catalogue_file = './Data/BigMDPL_RockstarHalo_z1.0.txt'
catalogue = CSVCatalog(catalogue_file,
                       ['mass', 'pid', 'x', 'y', 'z', 'vx', 'vy', 'vz']
                       )
catalogue['Position'] = np.array([catalogue['x'],
                                  catalogue['y'],
                                  catalogue['z']
                                  ]).T

boxsize = 2500
nmesh = 256

cm_original = catalogue.to_mesh(BoxSize=boxsize, Nmesh=nmesh, compensated=True)

catalogue['Selection'] = select_decide(catalogue.compute(catalogue['x']),
                                       0, boxsize
                                       )

cm_shuffled = catalogue.to_mesh(BoxSize=boxsize, Nmesh=nmesh, compensated=True)

power1d_original = FFTPower(cm_original, mode='1d').power
power1d_shuffled = FFTPower(cm_shuffled, mode='1d').power

k_orig, Pk_orig = power1d_original['k'], power1d_original['power'] - power1d_original.attrs['shotnoise']
k_shuf, Pk_shuf = power1d_shuffled['k'], power1d_shuffled['power'] - power1d_shuffled.attrs['shotnoise']

plt.style.use(style.mplstyle)
plt.figure('Power 1-d comparison', figsize=(8,6))
plt.loglog(k_orig, Pk_orig.real, '-k', marker='+', label='original')
plt.loglog(k_shuf, Pk_shuf.real, '--r', marker='x', label='shuffled')
plt.legend()
plt.xlabel(r'$k$ [$h/\text{Mpc}$]')
plt.ylabel(r'$P(k, z=1.0)$ [$(\text{Mpc}/h)^3$]')
plt.savefig('./Output/power1dcompare.pdf', format='pdf')
