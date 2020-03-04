"""Configuration for interactive notebooks.

"""
import os
import sys

import matplotlib as mpl
import seaborn as sns

current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "".join([current_file_dir, "/../"]))

from config import config

STYLESHEET = getattr(config, 'STYLESHEET')
PATH = getattr(config, 'DATAPATH')

mpl.pyplot.style.use(STYLESHEET)
sns.set(style='ticks', font='serif')

os.environ['MPLBACKEND'] = 'AGG'
