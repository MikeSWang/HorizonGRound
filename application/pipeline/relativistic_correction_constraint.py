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

# pylint: disable=no-name-in-module
from conf import PATHOUT, logger
from horizonground.clustering_modification import relativistic_correction_value

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

    parser.add_argument('--redshift', type=float)
    parser.add_argument('--chain-subdir', type=str, default='')
    parser.add_argument('--chain-file', type=str, default=None)
    parser.add_argument(
        '--contribution',
        choices=['all', 'evolution', 'magnification'], default='all'
    )

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
    chain_file = PATHOUT/progrc.chain_subdir/progrc.chain_file
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
    correction : list of float
        Relativistic correction value.

    """
    if progrc.contribution == 'all':
        correction = relativistic_correction_value(
            progrc.redshift,
            evolution_bias=biases[0], magnification_bias=biases[1]
        )
        return [correction]

    if progrc.contribution == 'evolution':
        correction = relativistic_correction_value(
            progrc.redshift,
            geometric=False, evolution_bias=biases[0], magnification_bias=None
        )
        return [correction]

    if progrc.contribution == 'magnification':
        correction = relativistic_correction_value(
            progrc.redshift,
            geometric=False, evolution_bias=None, magnification_bias=biases[1]
        )
        return [correction]

    raise ValueError("Which relativistic correction(s) unspecified.")


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
    :class:`pathlib.Path`
        Chain output file path.

    """
    infile = PATHOUT/progrc.chain_subdir/progrc.chain_file

    redshift_tag = "z{}".format(progrc.redshift)
    redshift_tag = redshift_tag if "." not in redshift_tag \
        else redshift_tag.rstrip("0")

    if progrc.contribution == 'all':
        prefix = "relcrct_"
    elif progrc.contribution == 'evolution':
        prefix = "relcrct_evol_"
    elif progrc.contribution == 'magnification':
        prefix = "relcrct_magn_"
    if redshift_tag not in progrc.chain_file:
        prefix += redshift_tag + "_"

    outfile = PATHOUT/progrc.chain_subdir/(
        prefix + progrc.chain_file
    ).replace("relbias_", "")

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

    LEVELS = [0.6826895, 0.9544997]
    QUANTILES = [0.1587, 0.5, 0.8413]
    COLOUR = '#A3C1AD'
    CORNER_OPTIONS = dict(
        color=COLOUR,
        quantiles=QUANTILES,
        levels=LEVELS,
        labels=_labels,
        label_kwargs={'visible': False},
        title_fmt='.5f',
        plot_datapoints=False,
        plot_contours=True,
        fill_contours=True,
        quiet=True,
        range=(0.999,)*len(_labels),
        rasterized=True,
        show_titles=True,
    )

    plt.close('all')

    distribution_fig = corner.corner(
        chain, bins=160, smooth=.75, smooth1d=.95, **CORNER_OPTIONS
    )

    fig_file = str(output_path).replace('.h5', '.pdf')
    if SAVEFIG:
        distribution_fig.savefig(fig_file)
    logger.info(
        "Saved distribution plot of relativistic correction samples.\n"
    )

    return distribution_fig


SAVE = True
SAVEFIG = True
if __name__ == '__main__':

    progrc = initialise()

    input_chain = load_samples()

    with Pool() as mpool:
        distilled_chain = distill_corrections(input_chain, pool=mpool)

    output_path = save_distilled()

    figures = view_distilled(distilled_chain)
