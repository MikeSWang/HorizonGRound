#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# style.py: VISUALISATION STYLE GUIDE
#
# Author: MS Wang
# Created: 2019-03
# =============================================================================

""":mod:`style`---visualisation style guide"""

# =============================================================================
# EXECUTION
# =============================================================================

mplstyle = {
    # Text
    'text.usetex': True,
    #'text.latex.preamble': [r'\usepackage{upgreek}'],

    # Font
    'font.family': 'serif',
    'font.size': 11.0,

    # Axes
    'axes.linewidth': 0.15,
    'axes.edgecolor': 'k',
    'axes.facecolor': 'w',
    'axes.labelsize': 'large',

    # Line styles
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'lines.antialiased': True,

    # Ticks
    'xtick.major.size': 5,
    'xtick.minor.size': 2.5,
    'xtick.major.width': 0.15,
    'xtick.minor.width': 0.15,
    'xtick.major.pad': 5,
    'xtick.minor.pad': 5,
    'xtick.labelsize': 'medium',
    'xtick.direction': 'in',
    'xtick.top': True,

    'ytick.major.size': 5,
    'ytick.minor.size': 2.5,
    'ytick.major.width': 0.15,
    'ytick.minor.width': 0.15,
    'ytick.major.pad': 5,
    'ytick.minor.pad': 5,
    'ytick.labelsize': 'medium',
    'ytick.direction': 'in',
    'ytick.right': True,

    # Legend
    'legend.fancybox': False,
    'legend.frameon' : False,
    'legend.fontsize': 'large',
    'legend.loc': 'best',

    # Figure
    'figure.figsize': [8,4.5],
    'figure.titlesize': 'large',

    # Saving figures
    'savefig.bbox': 'tight',
    'savefig.format': 'pdf',
    'savefig.transparent': True
}
