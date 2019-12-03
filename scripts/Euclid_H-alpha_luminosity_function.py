#!/usr/bin/env python
# coding: utf-8

# # *``Euclid``* H&alpha; Emitter Luminosity Function

# Initiate notebook.

# In[1]:


from collections import OrderedDict
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.cosmology import Planck15
from mpl_toolkits import mplot3d

from config import use_local_package

use_local_package("../../HorizonGRound/")

from horizonground.luminosity import (
    LuminosityFunctionModeller,
    alpha_emitter_luminosity_schechter_model,
)


# Define parameter ranges.

# In[2]:


REDSHIFT_RANGE = 0., 3.
LOG10_LUMINOSITY_RANGE = 39.0, 44.0
DENSITY_RANGE = 10**(-6), 10**(-1)


# ## Luminosity function measurements

# Determine redshift bins.

# In[3]:


redshift_bins = OrderedDict(
    zip(
        [r"0 < z < 0.3", r"0.3 < z < 0.6", r"0.6 < z < 0.9", 
         r"0.9 < z < 1.3", r"1.3 < z < 1.7", r"1.7 < z < 2.3"],
        [0.15, 0.45, 0.75, 1.1, 1.5, 2.0]
    )
)


# ## Luminosity function model

# Specify luminosity function model.

# In[4]:


modeller = LuminosityFunctionModeller.from_parameters_file(
    parameter_file="../data/input/Schechter_model_fits.txt",
    luminosity_model=alpha_emitter_luminosity_schechter_model,
    luminosity_variable='luminosity',
    threshold_value=3e-16,
    threshold_variable='flux',
    cosmology=Planck15
)

luminosity_function_model = modeller.luminosity_function


# Visualise luminosity function surface.

# In[5]:


NUM_MESH = 100

luminosities = np.linspace(*LOG10_LUMINOSITY_RANGE, num=NUM_MESH, endpoint=False)
redshifts = np.linspace(*REDSHIFT_RANGE, num=NUM_MESH, endpoint=False)
L, z = np.meshgrid(luminosities, redshifts)
Phi = luminosity_function_model(L, z)

fig = plt.figure("Luminosity function surface", figsize=(12, 7.75))
ax = plt.axes(projection='3d')

ax.plot_surface(
    L, z, Phi,
    cmap=sns.cubehelix_palette(as_cmap=True),
    edgecolor='none'
)

ax.set_xlabel(r"$\lg{L}$", fontsize=16)
ax.set_ylabel(r"$z$", fontsize=16)

ax.set_zlim(*DENSITY_RANGE)
ax.set_zscale('log')
ax.set_zlabel(r"$\Phi(\lg{L},z)$ [$\textrm{Mpc}^{-3}$]", fontsize=16)

