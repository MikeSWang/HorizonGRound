"""Configuration for executable scripts.

"""
import os
import sys

import matplotlib as mpl
import seaborn as sns

current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "".join([current_file_dir, "/../"]))

from config import program

STYLESHEET = getattr(program, 'STYLESHEET')
DATAPATH = getattr(program, 'DATAPATH')
PATHEXT = DATAPATH/"external"
PATHIN = DATAPATH/"input"
PATHOUT = DATAPATH/"output"

sns.set(style='ticks', font='serif')
mpl.pyplot.style.use(STYLESHEET)

os.environ['OMP_NUM_THREADS'] = '1'

sci_notation = getattr(program, 'sci_notation')
logger = program.setup_logger()
