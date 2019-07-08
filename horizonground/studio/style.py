# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# studio/style.py: VISUALISATION STYLE
#
# Copyright (C) 2019, MS Wang
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

""":mod:`~horizonground.studio.style` provides the visualisation style for use
with :mod:`matplotlib`.  User can modify default options in
:const:`horizon_style`.

"""

from cycler import cycler

horizon_style = {
    'figure.figsize': [8, 4.5],
    'figure.titlesize': 'large',

    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.transparent': True,
    # 'savefig.facecolor': True,
    # 'savefig.edgecolor': True,

    'text.usetex': True,
    # 'text.latex.preamble': [r'\usepackage{upgreek}'],
    'font.family': 'serif',
    'font.size': 11.0,

    'legend.fancybox': False,
    'legend.frameon': False,
    'legend.fontsize': 'large',
    'legend.loc': 'best',

    'axes.linewidth': 0.15,
    'axes.grid': False,
    # 'axes.grid.axis': 'both',
    # 'axes.autolimit_mode': 'round_numbers',
    # 'axes.xmargin': 0,
    # 'axes.ymargin': 0,
    'axes.edgecolor': 'k',
    'axes.facecolor': 'w',
    'axes.labelsize': 'large',
    'axes.prop_cycle': cycler(
            'color', ['#000000', '#C40233', '#0087BD', '#FFD300', '#009F6B',]
            ),  # Natural Colour System (NCS))

    'grid.alpha': 0.25,
    'grid.color': 'k',
    'grid.linewidth': 0.5,
    'grid.linestyle': (20, 4),

    'lines.linewidth': 2,
    'lines.markersize': 6,
    'lines.antialiased': True,

    'errorbar.capsize': 2.5,

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
    }
