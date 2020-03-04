r"""Relativistic correction constraint from sampled relativistic biases.

"""
import sys
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pprint import pformat

import corner
import h5py as hp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from conf import PATHOUT, logger
from horizonground.clustering_modification import relativistic_correction_eval

ORDERS = [0, 2]
LABELS = [r'$g(z={:.1f})$']


def initialise():
    """Initialise program.

    Returns
    -------
    program_configuration : :class:`argparse.Namespace`
        Parsed program configuration parameters.

    """
    parser = ArgumentParser("relativistic-correction-constraint")

    parser.add_argument('--redshift', type=float, default=2.)
    parser.add_argument('--chain-file', type=str, default=None)

    program_configuration = parser.parse_args()

    logger.info(
        "\n---Program configuration---\n%s\n",
        pformat(vars(program_configuration))
    )

    return program_configuration


def load_samples():
    """Load samples of relativistic biases from file.

    Returns
    -------
    bias_samples : :class:`numpy.ndarray`
        Relativistic bias samples.

    """
    chain_file = (PATHOUT/progrc.chain_file).with_suffix('.h5')
    with hp.File(chain_file, 'r') as chain_data:
        bias_samples = chain_data['extract/chain'][()]

    logger.info("Loaded bias samples from file: %s.\n", chain_file)

    return bias_samples


def compute_correction_from_biases(biases):
    """Compute relativistic correction function values from relativistic
    biases.

    Parameters
    ----------
    biases : :class:`numpy.ndarray`
        Evolution and magnification biases.

    Returns
    -------
    correction : [float]
        Relativistic correction value.

    """
    correction = relativistic_correction_eval(
        progrc.redshift, evolution_bias=biases[0], magnification_bias=biases[1]
    )

    return [correction]


def distill_corrections(bias_chain, pool=None):
    """Distill relativistic corrections from a relativistic bias chain.

    Parameters
    ----------
    bias_chain : :class:`numpy.ndarray`
        Relativistic bias parameter chain.
    pool : :class:`multiprocessing.Pool` or None, optional
        Multiprocessing pool (default is `None`).

    Returns
    -------
    correction_chain : :class:`numpy.ndarray`
        Relativistic correction samples.

    """
    mapping = pool.imap if pool else map
    num_cpus = cpu_count() if pool else 1

    logger.info(
        "Distilling relativistic corrections with %i CPUs...\n", num_cpus
    )
    correction_chain = list(
        tqdm(
            mapping(compute_correction_from_biases, bias_chain),
            total=len(bias_chain), mininterval=1, file=sys.stdout
        )
    )
    logger.info("... finished.\n")

    correction_chain = np.asarray(correction_chain)

    return correction_chain


def save_distilled():
    """Save the distilled relativistic correction chain.

    Returns
    -------
    outpath : :class:`pathlib.Path`
        Chain output file path.

    """
    infile = (PATHOUT/progrc.chain_file).with_suffix('.h5')

    redshift_tag = "z{}".format(progrc.redshift)
    redshift_tag = redshift_tag if "." not in redshift_tag \
        else redshift_tag.rstrip("0")

    if redshift_tag not in progrc.chain_file:
        prefix = "relcrct_" + redshift_tag + "_"
    else:
        prefix = "relcrct_"

    outfile = (PATHOUT/(prefix + progrc.chain_file)).with_suffix('.h5')

    with hp.File(infile, 'r') as indata, hp.File(outfile, 'w') as outdata:
        outdata.create_group('distill')
        try:
            indata.copy('extract/log_prob', outdata['distill'])
        except KeyError:
            pass
        outdata.create_dataset('distill/chain', data=distilled_chain)

    logger.info("Distilled chain saved to %s.\n", outfile)

    return outfile


def view_distilled(chain):
    """View the extracted chain of relativistic biases.

    Parameters
    ----------
    chain : :class:`numpy.ndarray`
        Chain.

    Returns
    -------
    distribution_fig : :class:`matplotlib.figure.Figure`
        Distribution figure.

    """
    _labels = [lab.format(progrc.redshift) for lab in LABELS]

    QUANTILES = [0.1587, 0.5, 0.8413]
    LEVELS = [0.39346934, 0.86466472]
    COLOUR = "#A3C1AD"
    CORNER_OPTIONS = dict(
        color=COLOUR,
        fill_contours=True,
        labels=_labels,
        label_kwargs={'visible': False},
        levels=LEVELS,
        plot_datapoints=False,
        plot_contours=True,
        quantiles=QUANTILES,
        quiet=True,
        rasterized=True,
        show_titles=True,
        title_fmt='.5f'
    )

    plt.close('all')

    distribution_fig = corner.corner(
        chain, bins=160, smooth=.75, smooth1d=.95, **CORNER_OPTIONS
    )

    if SAVEFIG:
        distribution_fig.savefig(output_path.with_suffix('.pdf'), format='pdf')
    logger.info(
        "Saved distribution plot of relativistic correction samples.\n"
    )

    return distribution_fig


SAVE = True
SAVEFIG = True
if __name__ == '__main__':

    progrc = initialise()

    input_chain = load_samples()

    with Pool() as pool:
        distilled_chain = distill_corrections(input_chain, pool=pool)

    output_path = save_distilled()

    figures = view_distilled(distilled_chain)
