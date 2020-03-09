"""Configuration for interactive notebooks.

"""
import os
import sys

import matplotlib as mpl
import seaborn as sns

current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "".join([current_file_dir, "/../"]))

from config import program

STYLESHEET = getattr(program, 'STYLESHEET')
PATH = getattr(program, 'DATAPATH')

sns.set(style='ticks', font='serif')
mpl.pyplot.style.use(STYLESHEET)
mpl.rcParams['font.size'] = 16

os.environ['MPLBACKEND'] = 'AGG'
