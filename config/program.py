"""Program configuration for applications and notebooks.

This modifies Python search path, provides custom logging facility and
``matplotlib`` style sheet, and sets up I/O paths.

"""
import logging
import os
import pathlib
import sys
import time

import matplotlib as mpl


class CustomLoggingFormatter(logging.Formatter):
    """Custom logging formatter.

    """

    start_time = time.time()

    def format(self, record):

        elapsed_time = record.created - self.start_time
        h, remainder_time = divmod(elapsed_time, 3600)
        m, s = divmod(remainder_time, 60)

        record.elapsed = "(+{}:{:02d}:{:02d})".format(int(h), int(m), int(s))

        return logging.Formatter.format(self, record)


def setup_logger():
    """Return the root logger suitably handled and formatted.

    Returns
    -------
    root_logger : :class:`logging.Logger`
        Root logger.

    """
    custom_formatter = CustomLoggingFormatter(
        fmt='[%(asctime)s %(elapsed)s %(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(custom_formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)

    return root_logger


def sci_notation(num):
    """Format integers in scientific notation.

    Parameters
    ----------
    num : int
        Integer to be formatted.

    Returns
    -------
    num_str : str
        Formatted string in scientific notation.

    """
    base, index = "{:.1e}".format(num).split("e")
    base = base.rstrip(".0").replace(".", ",")
    index = index.lstrip("+").lstrip("-").lstrip("0")

    num_str = "E".join([base, index])

    return num_str


# Configure logging.
logging.captureWarnings(True)

# Set I/O paths.
config_dir = os.path.dirname(os.path.abspath(__file__))

DATAPATH = pathlib.Path(config_dir).parent/"data"
STYLESHEET = mpl.rc_params_from_file(
    config_dir+"/horizon.mplstyle", use_default_template=False
)

# Modifies Python search path.
sys.path.append("".join([config_dir, "/../horizonground/"]))
