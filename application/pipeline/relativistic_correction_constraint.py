r"""Relativistic correction constraint from sampled relativistic biases.

"""
import sys
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pprint import pformat

import h5py as hp
import numpy as np
from nbodykit.cosmology import Planck15
from tqdm import tqdm

# pylint: disable=no-name-in-module
from conf import PATHOUT, logger
from horizonground.clustering_modification import (
    relativistic_correction_factor,
    relativistic_correction_value,
)


def initialise():
    """Initialise program.

    Returns
    -------
    program_configuration : :class:`argparse.Namespace`
        Parsed program configuration parameters.

    """
    parser = ArgumentParser("relativistic-correction-constraint")

    parser.add_argument('--redshift', type=float)
    parser.add_argument('--wavenumber', type=float)
    parser.add_argument('--chain-subdir', type=str, default='')
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
    chain_file = PATHOUT/progrc.chain_subdir/progrc.chain_file
    with hp.File(chain_file, 'r') as chain_data:
        bias_samples = chain_data['extract/chain'][()]

    logger.info("Loaded bias samples from file: %s.\n", chain_file)

    return bias_samples


def compute_corrections_from_biases(biases):
    """Compute relativistic correction values from relativistic biases.

    Parameters
    ----------
    biases : :class:`numpy.ndarray`
        Evolution and magnification biases.

    Returns
    -------
    list of float
        Relativistic correction values.

    """
    g_1 = relativistic_correction_value(
        progrc.redshift, 1,
        evolution_bias=biases[0], magnification_bias=biases[1]
    )
    g_2 = relativistic_correction_value(
        progrc.redshift, 2,
        evolution_bias=biases[0], magnification_bias=biases[1]
    )

    return [g_1, g_2]


def compute_factors_from_corrections(corrections):
    """Compute relativistic correction factors from relativistic
    correction values.

    Parameters
    ----------
    corrections : :class:`numpy.ndarray`
        Relativistic corrections.

    Returns
    -------
    list of float
        Relativistic correction factor values.

    """
    delta_P_0 = relativistic_correction_factor(
        progrc.wavenumber, 0, progrc.redshift, b_1,
        correction_value_1=corrections[0], correction_value_2=corrections[1]
    )
    delta_P_2 = relativistic_correction_factor(
        progrc.wavenumber, 2, progrc.redshift, b_1,
        correction_value_1=corrections[0], correction_value_2=corrections[1]
    )

    return [delta_P_0, delta_P_2]


def distill_corrections(bias_chain, pool=None):
    """Distill relativistic corrections and correction factors from a
    relativistic bias chain.

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
    factor_chain : :class:`numpy.ndarray`
        Relativistic correction factor samples.

    """
    mapping = pool.imap if pool else map
    num_cpus = cpu_count() if pool else 1

    logger.info(
        "Distilling relativistic corrections/correction factors "
        + "with %i CPUs...\n", num_cpus
    )

    correction_chain = np.asarray(list(
        tqdm(
            mapping(compute_corrections_from_biases, bias_chain),
            total=len(bias_chain), mininterval=1, file=sys.stdout
        )
    ))

    factor_chain = list(
        tqdm(
            mapping(compute_factors_from_corrections, correction_chain),
            total=len(correction_chain), mininterval=1, file=sys.stdout
        )
    )

    logger.info("... finished.\n")

    return correction_chain, factor_chain


def save_distilled():
    """Save the distilled relativistic correction and correction factor
    chains.

    Returns
    -------
    :class:`pathlib.Path`
        Chain output file path.

    """
    infile = PATHOUT/progrc.chain_subdir/progrc.chain_file

    chain_suffix = progrc.chain_file.replace("relbias_", "")

    redshift_tag = "z{:.2f}".format(progrc.redshift)

    prefix_0, prefix_1 = "relcrct_", "relfact_"
    if redshift_tag not in progrc.chain_file:
        prefix_0 += redshift_tag + "_"
        prefix_1 += redshift_tag + "_k{}_".format(progrc.wavenumber)

    outfile_0 = PATHOUT/progrc.chain_subdir/(prefix_0 + chain_suffix)
    outfile_1 = PATHOUT/progrc.chain_subdir/(prefix_1 + chain_suffix)

    with hp.File(infile, 'r') as indata, hp.File(outfile_0, 'w') as outdata:
        outdata.create_group('distill')
        try:
            indata.copy('extract/log_prob', outdata['distill'])
        except KeyError:
            pass
        outdata.create_dataset('distill/chain', data=distilled_chains[0])

    with hp.File(infile, 'r') as indata, hp.File(outfile_1, 'w') as outdata:
        outdata.create_group('distill')
        try:
            indata.copy('extract/log_prob', outdata['distill'])
        except KeyError:
            pass
        outdata.create_dataset('distill/chain', data=distilled_chains[1])

    logger.info("Distilled chains saved to %s and %s.\n", outfile_0, outfile_1)


SAVE = True
SAVEFIG = True
if __name__ == '__main__':

    progrc = initialise()

    b_1 = 1.2 / Planck15.scale_independent_growth_factor(progrc.redshift)

    input_chain = load_samples()

    with Pool() as mpool:
        distilled_chains = distill_corrections(input_chain, pool=mpool)

    save_distilled()
