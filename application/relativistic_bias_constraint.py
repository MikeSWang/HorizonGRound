r"""Relativistic bias constraint from sampled luminosity function model.

"""
import sys
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pprint import pformat

import corner
import emcee as mc
import h5py as hp
import matplotlib.pyplot as plt
import numpy as np
from astropy import cosmology
from tqdm import tqdm

from conf import PATHOUT, logger
from horizonground.lumfunc_modeller import (
    LumFuncModeller,
    konstante_correction
)
import horizonground.lumfunc_modeller as lumfunc_modeller

LABELS = [r'$f_\mathrm{{e}}(z={})$', r'$s(z={})$']
NDIM = len(LABELS)

burnin, reduce = 0, 1


def initialise():
    """Initialise program.

    Returns
    -------
    program_configuration : :class:`argparse.Namespace`
        Parsed program configuration parameters.

    """
    parser = ArgumentParser("relativistic-bias-constraint")

    parser.add_argument('--model-name', type=str, default='quasar_PLE')
    parser.add_argument('--redshift', type=float, default=2.)

    parser.add_argument('--sampler', type=str.lower, choices=['emcee', 'zeus'])
    parser.add_argument('--chain-file', type=str, default=None)
    parser.add_argument('--burnin', type=int, default=None)
    parser.add_argument('--reduce', type=int, default=None)

    program_configuration = parser.parse_args()

    logger.info(
        "\n---Program configuration---\n%s\n",
        pformat(vars(program_configuration)).lstrip("{").rstrip("}")
    )

    return program_configuration


def read_chains():
    """Load and process chains from file.

    Returns
    -------
    flat_chain : :class:`numpy.ndarray`
        Flattened chains.

    """
    # Read chains into memory.
    chain_file = PATHOUT/progrc.chain_file

    if progrc.sampler == 'emcee':
        reader = mc.backends.HDFBackend(chain_file, read_only=True)
    elif progrc.sampler == 'zeus':
        chain_data = hp.File(chain_file, 'r')
        reader = chain_data['mcmc']

    logger.info("Loaded chain file: %s.\n", chain_file)

    # Process chains by burn-in and thinning.
    if progrc.burnin is None or progrc.reduce is None:
        try:
            autocorr_time = reader.get_autocorr_time()
        except AttributeError:
            autocorr_time = reader['autocorr_time'][()]
        except mc.autocorr.AutocorrError as warning:
            logger.warning(warning)
            autocorr_time = None

    if progrc.burnin is None:
        try:
            _burnin = 4 * int(np.max(autocorr_time))  # can change 4 to 2
        except (TypeError, ValueError):
            _burnin = 0
    else:
        _burnin = progrc.burnin

    if progrc.reduce is None:
        try:
            _reduce = int(np.min(autocorr_time)) // 5  # can change 5 to 2
        except (TypeError, ValueError):
            _reduce = 1
    else:
        _reduce = progrc.reduce

    # Flatten chains.
    if progrc.sampler == 'emcee':
        flat_chain = reader.get_chain(flat=True, discard=_burnin, thin=_reduce)
    elif progrc.sampler == 'zeus':
        flat_chain = reader['chain'][_burnin::_reduce, :, :]\
            .reshape((-1, len(PARAMETERS)))
        chain_data.close()

    logger.info(
        "Chain flattened with %i burn-in and %i thinning.\n", _burnin, _reduce
    )

    return flat_chain, _burnin, _reduce


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
    lumfunc_model = getattr(lumfunc_modeller, progrc.model_name + '_lumfunc')

    model_parameters = dict(zip(PARAMETERS, lumfunc_params))

    modeller = LumFuncModeller(
        lumfunc_model, model_parameters,
        LUMINOSITY_VARIABLE, threshold_value, COSMOLOGY
    )

    bias_evo = modeller.evolution_bias(progrc.redshift)
    bias_mag = modeller.magnification_bias(progrc.redshift)

    return bias_evo, bias_mag


def extract_biases(lumfunc_param_chain, pool=None):
    """Extract relativistic biases from a luminosity function parameter
    chain.

    Parameters
    ----------
    lumfunc_model_chain : :class:`numpy.ndarray`
        Luminosity function parameter chain.
    pool : :class:`multiprocessing.Pool` *or None, optional*
        Multiprocessing pool (default is `None`).

    Returns
    -------
    bias_samples : :class:`numpy.ndarray`
        Relativistic bias samples.

    """
    mapping = pool.imap if pool else map
    num_cpus = cpu_count() if pool else 1

    logger.info("Resampling relativistic biases with %i CPUs...\n", num_cpus)
    bias_samples = list(tqdm(
        mapping(compute_biases_from_lumfunc, lumfunc_param_chain),
        total=len(lumfunc_param_chain), mininterval=15, file=sys.stdout
    ))
    logger.info("... finished.\n")

    bias_samples = np.asarray(bias_samples)

    return bias_samples


def save_extracts():
    """Save extracted relativistic bias chains.

    Returns
    -------
    :class:`pathlib.Path`
        Chain output file path.

    """
    infile = PATHOUT/progrc.chain_file

    redshift_tag = "z{}".format(progrc.redshift)
    redshift_tag = redshift_tag if "." not in redshift_tag \
        else redshift_tag.rstrip("0")

    prefix = "relbias_" + redshift_tag + "_"

    outfile = PATHOUT/(prefix + progrc.chain_file)

    with hp.File(infile, 'r') as indata, hp.File(outfile, 'w') as outdata:
        outdata.create_group('extract')
        outdata.create_dataset('extract/chain', data=extracted_chain)
        if progrc.sampler == 'emcee':
            outdata.create_dataset(
                'extract/log_prob',
                data=np.ravel(indata['mcmc/log_prob'][burnin::reduce, :])
            )

    logger.info("Extracted chain saved to %s.\n", outfile)

    return outfile


def load_extracts(chain_file):
    """Load extracted relativistic bias chains.

    Parameters
    ----------
    chain_file : :class:`pathlib.Path` *or str*
        Extracted relativistic bias chain file.

    Returns
    -------
    extracts : :class:`numpy.ndarray`
        Relativistic bias samples.

    """
    with hp.File(PATHOUT/chain_file, 'r') as chain_data:
        extracts = chain_data['extract/chain'][()]

    return extracts


def view_extracts(chain):
    """View the extracted chain of relativistic biases.

    Parameters
    ----------
    chain : :class:`numpy.ndarray`
        Chain.

    Returns
    -------
    chain_fig, contour_fig : :class:`matplotlib.figure.Figure`
        Chain and contour figures.

    """
    LEVELS = [0.39346934, 0.86466472]
    QUANTILES = [0.1587, 0.5, 0.8413]
    COLOUR = '#A3C1AD'
    CORNER_OPTIONS = dict(
        color=COLOUR,
        quantiles=QUANTILES,
        levels=LEVELS,
        labels=[lab.format(progrc.redshift) for lab in LABELS],
        label_kwargs={'visible': False},
        plot_datapoints=False,
        plot_contours=True,
        fill_contours=True,
        range=(0.999,)*NDIM,
        show_titles=True,
        title_fmt='.3f',
        quiet=True,
        rasterized=True,
    )

    plt.close('all')

    chain_fig, axes = plt.subplots(NDIM, figsize=(12, NDIM), sharex=True)
    for param_idx in range(NDIM):
        ax = axes[param_idx]
        ax.plot(
            chain[:, param_idx], color=COLOUR, alpha=0.66, rasterized=True
        )
        ax.set_xlim(0, len(chain))
        ax.set_ylabel(LABELS[param_idx])
    axes[-1].set_xlabel("steps")

    fig_file = str(output_path).replace('.h5', '.chains.pdf')
    if SAVEFIG:
        chain_fig.savefig(fig_file)
    logger.info("Saved chain plot of relativistic bias samples.\n")

    contour_fig = corner.corner(
        chain, bins=160, smooth=.75, smooth1d=.95, **CORNER_OPTIONS
    )

    fig_file = str(output_path).replace('.h5', '.contours.pdf')
    if SAVEFIG:
        contour_fig.savefig(fig_file)
    logger.info("Saved contour plot of relativistic bias samples.\n")

    return contour_fig


# Model-independent settings.
# pylint: disable=no-member
BASE10_LOG = True
COSMOLOGY = cosmology.Planck15

# Model-specific settings.
PARAMETERS = [
    'm_\\ast(z_\\mathrm{p})', '\\lg\\Phi_\\ast',
    '\\alpha_\\mathrm{l}', '\\alpha_\\mathrm{h}',
    '\\beta_\\mathrm{l}', '\\beta_\\mathrm{h}',
    'k_{1\\mathrm{l}}', 'k_{1\\mathrm{h}}',
    'k_{2\\mathrm{l}}', 'k_{2\\mathrm{h}}',
]

LUMINOSITY_VARIABLE = 'magnitude'
THRESHOLD_VARIABLE = 'magnitude'

# Program-specific settings.
SAVE = True
SAVEFIG = True

if __name__ == '__main__':

    progrc = initialise()

    input_chain, burin, reduce = read_chains()  #

    threshold_value = 22.5 - konstante_correction(progrc.redshift)

    with Pool() as mpool:  #
        extracted_chain = extract_biases(input_chain, pool=mpool)  #

    output_path = save_extracts()  #
    # extracted_chain = load_extracts(progrc.chain_file)  #

    figures = view_extracts(extracted_chain)
