r"""Relativistic bias constraint from sampled luminosity function model.

"""
from argparse import ArgumentParser
from pprint import pformat

import corner
import emcee as mc
import matplotlib.pyplot as plt
import numpy as np
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
    flat_chain :
        Flattened chains.

    """
    chain_file = PATHOUT/progrc.chain_file

    # Read chains into memory.
    if chain_file.suffix == '.h5':
        reader = mc.backends.HDFBackend(chain_file, read_only=True)
    elif chain_file.suffix == '.npy':
        pass  # FIXME
    logger.info("Loaded chain file: %s.\n", chain_file)

    # Process chains by burn-in and thinning.
    try:
        autocorr_time = reader.get_autocorr_time()
    except mc.autocorr.AutocorrError as act_warning:
        autocorr_time = None
        logger.warning(act_warning)

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

    flat_chain = reader.get_chain(flat=True, discard=burnin, thin=reduce)
    logger.info(
        "Chain flattened with %i burn-in and %i thinning.\n", burnin, reduce
    )

    return flat_chain


def sample_biases(lumfunc_model_chains):
    """Extract samples of relativistic biases from luminosity function
    parameter chains.

    Parameters
    ----------
    lumfunc_model_chains :
        Luminosity function parameter chains.

    Returns
    -------
    bias_samples :
        Relativistic bias samples.

    """
    lumfunc_model = getattr(lumfunc_modeller, progrc.model_name)

    modeller_args = (
        BRIGHTNESS_VARIABLE, THRESHOLD_VALUE, THRESHOLD_VARIABLE, COSMOLOGY
    )

    bias_samples = np.zeros((len(lumfunc_model_chains), len(LABELS)))
    for chain_step, chain_params in enumerate(lumfunc_model_chains):
        model_parameters = dict(zip(PARAMETERS, chain_params))
        modeller = LumFuncModeller(
            lumfunc_model, *modeller_args, base10_log=BASE10_LOG,
            **model_parameters
        )
        bias_samples[chain_step] = (
            modeller.evolution_bias(progrc.redshift),
            modeller.magnification_bias(progrc.redshift)
        )

    return bias_samples


def view_chain(chain):
    """Extract samples of relativistic biases from chains.

    Parameters
    ----------
    chain :
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
    rechain = sample_biases(inchain)
    figures = view_chain(rechain)
