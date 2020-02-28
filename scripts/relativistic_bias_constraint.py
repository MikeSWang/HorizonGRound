r"""Relativistic bias constraint from sampled luminosity function model.

"""
import os
import sys
import multiprocessing as mp
from argparse import ArgumentParser
from pprint import pformat

os.environ['OMP_NUM_THREADS'] = '1'

import corner
import emcee as mc
import h5py as hp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from astropy import cosmology

from config import PATHOUT, logger, use_local_package

use_local_package("../../HorizonGRound/")

import horizonground.lumfunc_modeller as lumfunc_modeller
from horizonground.lumfunc_modeller import LumFuncModeller

PARAMETERS = [
    'M_{g\\ast}(z_\\textrm{p})', '\\lg\\Phi_\\ast',
    '\\alpha_\\textrm{l}', '\\alpha_\\textrm{h}',
    '\\beta_\\textrm{l}', '\\beta_\\textrm{h}',
    'k_{1\\textrm{l}}', 'k_{1\\textrm{h}}',
    'k_{2\\textrm{l}}', 'k_{2\\textrm{h}}',
]
LABELS = [r'$f_\textrm{e}$', r'$s$']


def initialise():
    """Initialise program.

    Returns
    -------
    program_configuration : :class:`argparse.Namespace`
        Parsed program configuration parameters.

    """
    parser = ArgumentParser("relativistic-bias-constraint")

    parser.add_argument('--redshift', type=float, default=2.)

    parser.add_argument('--model-name', type=str, default='quasar_PLE_model')
    parser.add_argument('--chain-file', type=str, default=None)

    parser.add_argument('--burnin', type=int, default=None)
    parser.add_argument('--reduce', type=int, default=None)

    program_configuration = parser.parse_args()

    logger.info(
        "\n---Program configuration---\n%s\n",
        pformat(vars(program_configuration))
    )

    return program_configuration


def read_chains():
    """Load and process chains from files.

    Returns
    -------
    flat_chain : :class:`numpy.ndarray`
        Flattened chains.

    """
    chain_file = PATHOUT/progrc.chain_file

    # Read chains into memory.
    if chain_file.suffix == '.h5':
        reader = mc.backends.HDFBackend(chain_file, read_only=True)
    elif chain_file.suffix == '.npy':
        reader = np.load(chain_file).item()
    logger.info("Loaded chain file: %s.\n", chain_file)

    # Process chains by burn-in and thinning.
    if progrc.burnin is None and progrc.reduce is None:
        if chain_file.suffix == '.h5':
            try:
                autocorr_time = reader.get_autocorr_time()
            except mc.autocorr.AutocorrError as act_warning:
                autocorr_time = None
                logger.warning(act_warning)
        elif chain_file.suffix == '.npy':
            autocorr_time = reader['autocorr_time']

    if progrc.burnin is None:
        try:
            burnin = 4 * int(np.max(autocorr_time))  # can change 4 to 2
        except (TypeError, ValueError):
            burnin = 0
    else:
        burnin = progrc.burnin

    if progrc.reduce is None:
        try:
            reduce = int(np.min(autocorr_time)) // 5  # can change 5 to 2
        except (TypeError, ValueError):
            reduce = 1
    else:
        reduce = progrc.reduce

    if chain_file.suffix == '.h5':
        flat_chain = reader.get_chain(flat=True, discard=burnin, thin=reduce)
    elif chain_file.suffix == '.npy':
        flat_chain = np.swapaxes(reader['chain'], 0, 1)[burnin::reduce, :, :]\
            .reshape((-1, len(PARAMETERS)))
    logger.info(
        "Chain flattened with %i burn-in and %i thinning.\n", burnin, reduce
    )

    return flat_chain


def compute_biases_from_lumfunc(lumfunc_params):
    """Compute relativistic biases from the luminosity function model.

    Parameters
    ----------
    lumfunc_params : :class:`numpy.ndarray`
        Luminosity function model parameters.

    Returns
    -------
    bias_evo, bias_mag : :class:`numpy.ndarray`
        Relativistic evolution or magnification bias.

    """
    lumfunc_model = getattr(lumfunc_modeller, progrc.model_name)
    modeller_args = (
        BRIGHTNESS_VARIABLE, THRESHOLD_VALUE, THRESHOLD_VARIABLE, COSMOLOGY
    )

    model_parameters = dict(zip(PARAMETERS, lumfunc_params))

    modeller = LumFuncModeller(
        lumfunc_model, *modeller_args,
        base10_log=BASE10_LOG, **model_parameters
    )

    bias_evo = modeller.evolution_bias(progrc.redshift)
    bias_mag = modeller.magnification_bias(progrc.redshift)

    return bias_evo, bias_mag


def resample_biases(lumfunc_param_chains, pool=None):
    """Resample relativistic biases from luminosity function
    parameter chains.

    Parameters
    ----------
    lumfunc_model_chains : :class:`numpy.ndarray`
        Luminosity function parameter chains.
    pool : :class:`multiprocessing.Pool` or None, optional
        Multiprocessing pool (default is `None`).

    Returns
    -------
    bias_samples : :class:`numpy.ndarray`
        Relativistic bias samples.

    """
    mapping = pool.imap if pool else map
    ncpus = mp.cpu_count() if pool else 1

    logger.info("Resampling relativistic biases with %i CPUs...\n", ncpus)
    bias_samples = list(
        tqdm(
            mapping(compute_biases_from_lumfunc, lumfunc_param_chains),
            total=len(lumfunc_param_chains), mininterval=1, file=sys.stdout
        )
    )
    logger.info("\n... finished.\n")

    bias_samples = np.array(bias_samples)

    with hp.File(PATHOUT/progrc.chain_file, 'r') as indata, \
            hp.File(PATHOUT/("relbias_" + progrc.chain_file), 'w') as outdata:
        outdata.create_group('extract')
        indata.copy('mcmc/accepted', outdata['/extract'])
        indata.copy('mcmc/log_prob', outdata['/extract'])
        outdata.create_dataset('extract/rechain', data=bias_samples)

    return bias_samples


def load_rechain(rechain_file):
    """Load resampled relativistic bias chains.

    Parameters
    ----------
    rechain_file : :class:`pathlib.Path` or str
        Resampled relativistic bias chain file path inside ``PATHOUT/``.

    Returns
    -------
    rechain_samples : :class:`numpy.ndarray`
        Relativistic bias samples.

    """
    h5file = Path(PATHOUT/rechain_file).with_suffix('.h5')
    with hp.File(h5file, 'r') as chainfile:
        rechain_samples = chainfile['extract/rechain'][()]

    return rechain_samples


def view_chain(chain):
    """Extract samples of relativistic biases from chains.

    Parameters
    ----------
    chain : :class:`numpy.ndarray`
        Chain.

    Returns
    -------
    chain_fig, contour_fig : :class:`matplotlib.figure.Figure`
        Chain and contour figures.

    """
    COLOUR = "#A3C1AD"
    QUANTILES = [0.1587, 0.5, 0.8413]
    LEVELS = [0.39346934, 0.86466472]
    SAVEFIG = True

    ndim = len(LABELS)
    output_file = PATHOUT/progrc.chain_file
    corner_opt = dict(
        color=COLOUR,
        fill_contours=True,
        labels=LABELS,
        label_kwargs={'visible': False},
        levels=LEVELS,
        plot_datapoints=False,
        plot_contours=True,
        quantiles=QUANTILES,
        quiet=True,
        rasterized=True,
        show_titles=True,
    )

    plt.close('all')

    chain_fig, axes = plt.subplots(ndim, figsize=(12, ndim), sharex=True)
    for param_idx in range(ndim):
        ax = axes[param_idx]
        ax.plot(
            chain[:, param_idx], color=COLOUR, alpha=0.66, rasterized=True
        )
        ax.set_xlim(0, len(chain))
        ax.set_ylabel(LABELS[param_idx])
    axes[-1].set_xlabel("steps")

    if SAVEFIG:
        chain_fig.savefig(output_file.with_suffix('.chain.pdf'), format='pdf')
    logger.info("Saved chain plot of relativistic bias samples.\n")

    contour_fig = corner.corner(chain, bins=100, smooth=0.4, **corner_opt)

    if SAVEFIG:
        contour_fig.savefig(output_file.with_suffix('.pdf'), format='pdf')
    logger.info("Saved contour plot of relativistic bias samples.\n")

    return chain_fig, contour_fig


BRIGHTNESS_VARIABLE = 'magnitude'
THRESHOLD_VARIABLE = 'magnitude'
THRESHOLD_VALUE = -21.80
BASE10_LOG = True
COSMOLOGY = cosmology.Planck15

if __name__ == '__main__':
    progrc = initialise()
    inchain = read_chains()

    with mp.Pool() as mp_pool:
        rechain = resample_biases(inchain[8250000:], pool=mp_pool)

    figures = view_chain(rechain)
