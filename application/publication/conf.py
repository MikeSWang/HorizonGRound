"""Configuration for interactive notebooks.

"""
import os
import sys

import matplotlib as mpl
import seaborn as sns

try:
    from config import program
except ImportError:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, "".join([current_file_dir, "/../"]))
    from config import program

stylesheet = getattr(program, 'stylesheet')
data_dir = getattr(program, 'data_dir')

sns.set(style='ticks', font='serif')
mpl.pyplot.style.use(stylesheet)

os.environ['MPLBACKEND'] = 'AGG'
mpl.rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}'

PATHEXT = data_dir/"external"
PATHIN = data_dir/"input"
PATHOUT = data_dir/"output"
