"""Configuration for executable scripts.

"""
import os
import sys

import matplotlib as mpl
import seaborn as sns

current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "".join([current_file_dir, "/../"]))

from config import config

STYLESHEET = getattr(config, 'STYLESHEET')
DATAPATH = getattr(config, 'DATAPATH')
PATHEXT = DATAPATH/"external"
PATHIN = DATAPATH/"input"
PATHOUT = DATAPATH/"output"

mpl.pyplot.style.use(STYLESHEET)
sns.set(style='ticks', font='serif')

os.environ['OMP_NUM_THREADS'] = '1'

sci_notation = getattr(config, 'sci_notation')
logger = config.setup_logger()
