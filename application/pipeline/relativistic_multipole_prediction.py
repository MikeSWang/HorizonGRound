r"""Relativistic multipole predictions from sampled relativistic
correction values.

"""
import sys
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pprint import pformat

import corner
import h5py as hp
import numpy as np
from nbodykit.cosmology import Planck15
from tqdm import tqdm

# pylint: disable=no-name-in-module
from conf import PATHOUT, logger
from horizonground.clustering_modification import (
    relativistic_correction_factor,
)


def initialise():
    """Initialise program.

    Returns
    -------
    program_configuration : :class:`argparse.Namespace`
        Parsed program configuration parameters.

    """
    parser = ArgumentParser("relativistic-correction-factors")

    parser.add_argument('--redshift', type=float)
    parser.add_argument('--chain-subdir', type=str, default='')
    parser.add_argument('--chain-file', type=str, default=None)

    program_configuration = parser.parse_args()

    logger.info(
        "\n---Program configuration---\n%s\n",
        pformat(vars(program_configuration))
    )

    return program_configuration


def load_samples():
    """Load samples of relativistic corrections from file.

    Returns
    -------
    correction_samples : :class:`numpy.ndarray`
        Relativistic correction samples.

    """
    chain_file = PATHOUT/progrc.chain_subdir/progrc.chain_file

    with hp.File(chain_file, 'r') as chain_data:
        correction_samples = chain_data['distill/chain'][()]

    logger.info("Loaded correction samples from file: %s.\n", chain_file)

    return correction_samples


def compute_factors_from_corrections(corrections):
    """Compute relativistic correction factors from relativistic
    corrections.

    Parameters
    ----------
    corrections : :class:`numpy.ndarray`
        Relativistic corrections.

    Returns
    -------
    list of float
        Relativistic correction factors.

    """
    delta_P0 = np.array([
        relativistic_correction_factor(
            k, 0, progrc.redshift, b_1,
            correction_value_1=corrections[0],
            correction_value_2=corrections[1]
        )
        for k in wavenumbers
    ])
    delta_P2 = np.array([
        relativistic_correction_factor(
            k, 2, progrc.redshift, b_1,
            correction_value_1=corrections[0],
            correction_value_2=corrections[1]
        )
        for k in wavenumbers
    ])
    return [delta_P0, delta_P2]


def distill_factor_quantiles(correction_chain, pool=None):
    """Distill relativistic correction factor quantiles from a
    relativistic correction chain.

    Parameters
    ----------
    correction_chain : :class:`numpy.ndarray`
        Relativistic correction_chain chain.
    pool : :class:`multiprocessing.Pool` or None, optional
        Multiprocessing pool (default is `None`).

    Returns
    -------
    factor_quantiles : dict of dict of :class:`numpy.ndarray`
        Relativistic correction factor quantiles.

    """
    mapping = pool.imap if pool else map
    num_cpus = cpu_count() if pool else 1

    quantile_levels = [0.022750, 0.158655, 0.5, 0.841345, 0.977250]
    factor_quantiles = {0: defaultdict(list), 2: defaultdict(list)}

    logger.info(
        "Distilling relativistic correction factors at redshift %.2f "
        + "with %i CPUs...\n", progrc.redshift, num_cpus
    )

    factor_chain = np.asarray(list(
        tqdm(
            mapping(compute_factors_from_corrections, correction_chain),
            total=len(correction_chain), mininterval=1, file=sys.stdout
        )
    ))

    factor_0_q = np.asarray([
        corner.quantile(factor_chain[:, 0, k_idx], q=quantile_levels)
        for k_idx, k in enumerate(wavenumbers)
    ])
    factor_2_q = np.asarray([
        corner.quantile(factor_chain[:, 1, k_idx], q=quantile_levels)
        for k_idx, k in enumerate(wavenumbers)
    ])

    for q_idx, q in enumerate([-2, -1, 0, 1, 2]):
        factor_quantiles[0][q].append(factor_0_q[:, q_idx])
        factor_quantiles[2][q].append(factor_2_q[:, q_idx])

    logger.info("... finished.\n")

    return factor_quantiles


def save_distilled():
    """Save the distilled relativistic factor quantiles

    Returns
    -------
    :class:`pathlib.Path`
        Chain output file path.

    """
    prefix = "relpole_"
    redshift_tag = "z{:.2f}".format(progrc.redshift)
    chain_suffix = progrc.chain_file.replace("relbias_", "")

    if redshift_tag not in progrc.chain_file:
        prefix += redshift_tag + "_"

    outfile = PATHOUT/progrc.chain_subdir/(prefix + chain_suffix)
    with hp.File(outfile, 'w') as outdata:
        for ell in [0, 2]:
            for q in [-2, -1, 0, 1, 2]:
                outdata.create_dataset(
                    '{}/{}'.format(ell, q),
                    data=np.asarray(distilled_quantiles[ell][q])
                )

    logger.info("Distilled chains saved to %s.\n", outfile)


if __name__ == '__main__':

    progrc = initialise()

    b_1 = 1.2 / Planck15.scale_independent_growth_factor(progrc.redshift)
    wavenumbers = np.logspace(-4., -1., num=60+1)

    input_chain = load_samples()

    with Pool() as mpool:
        distilled_quantiles = distill_factor_quantiles(input_chain, pool=mpool)

    save_distilled()
