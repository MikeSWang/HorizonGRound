r"""Tracer number density from sampled luminosity function model.

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

LABELS = [r'$\bar{{n}}(z={})$']
NDIM = len(LABELS)

burnin, reduction = 0, 1


def initialise():
    """Initialise program.

    Returns
    -------
    program_configuration : :class:`argparse.Namespace`
        Parsed program configuration parameters.

    """
    parser = ArgumentParser("tracer-number-density")

    parser.add_argument('--task', type=str, choices=['extract', 'load'])
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--redshift', type=float)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--apparent_to_absolute', action='store_true')

    parser.add_argument('--sampler', type=str.lower, choices=['emcee', 'zeus'])
    parser.add_argument('--chain-file', type=str, default=None)
    parser.add_argument('--burnin', type=int, default=None)
    parser.add_argument('--reduction', type=int, default=1)

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
    if progrc.burnin is None or progrc.reduction == 0:
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

    if progrc.reduction == 0:
        try:
            _reduction = int(np.min(autocorr_time)) // 5  # can change 5 to 2
        except (TypeError, ValueError):
            _reduction = 1
    else:
        _reduction = progrc.reduction

    # Flatten chains.
    if progrc.sampler == 'emcee':
        flat_chain = reader.get_chain(
            flat=True, discard=_burnin, thin=_reduction
        )
    elif progrc.sampler == 'zeus':
        flat_chain = reader['chain'][_burnin::_reduction, :, :]\
            .reshape((-1, len(parameters)))
        chain_data.close()

    logger.info(
        "Chain flattened with %i burn-in and %i thinning.\n",
        _burnin, _reduction
    )

    return flat_chain, _burnin, _reduction


def compute_density_from_lumfunc(lumfunc_params):
    r"""Compute tracer number density from the luminosity function model.

    Parameters
    ----------
    lumfunc_params : :class:`numpy.ndarray`
        Luminosity function model parameters.

    Returns
    -------
    number_density : :class:`numpy.ndarray`
        Tracer number density (in cubic :math:`h`/Mpc).

    """
    lumfunc_model = getattr(lumfunc_modeller, progrc.model_name + '_lumfunc')

    model_parameters = dict(zip(parameters, lumfunc_params))

    modeller = LumFuncModeller(
        lumfunc_model, model_parameters,
        LUMINOSITY_VARIABLE, luminosity_threshold, COSMOLOGY
    )

    number_density = modeller.comoving_number_density(progrc.redshift)

    return number_density


def extract_number_density(lumfunc_param_chain, pool=None):
    """Extract tracer number density from a luminosity function parameter
    chain.

    Parameters
    ----------
    lumfunc_model_chain : :class:`numpy.ndarray`
        Luminosity function parameter chain.
    pool : :class:`multiprocessing.Pool` *or None, optional*
        Multiprocessing pool (default is `None`).

    Returns
    -------
    density_samples : :class:`numpy.ndarray`
        Tracer number density samples.

    """
    mapping = pool.imap if pool else map
    num_cpus = cpu_count() if pool else 1

    logger.info("Resampling tracer number density with %i CPUs...\n", num_cpus)
    density_samples = list(tqdm(
        mapping(compute_density_from_lumfunc, lumfunc_param_chain),
        total=len(lumfunc_param_chain), mininterval=15, file=sys.stdout
    ))
    logger.info("... finished.\n")

    density_samples = np.asarray(density_samples)

    return density_samples


def save_extracts():
    """Save extracted tracer number density chains.

    Returns
    -------
    :class:`pathlib.Path`
        Chain output file path.

    """
    infile = PATHOUT/progrc.chain_file

    _redshift_tag = "_z{:.2f}".format(progrc.redshift)
    _threshold_tag = "_m{:.1f}".format(luminosity_threshold)

    _prefix = "numden" + _redshift_tag + _threshold_tag + "_"

    outfile = PATHOUT/(_prefix + progrc.chain_file)

    with hp.File(infile, 'r') as indata, hp.File(outfile, 'w') as outdata:
        outdata.create_group('extract')
        outdata.create_dataset('extract/chain', data=extracted_chain)
        if progrc.sampler == 'emcee':
            outdata.create_dataset(
                'extract/log_prob',
                data=np.ravel(indata['mcmc/log_prob'][burnin::reduction, :])
            )

    logger.info("Extracted chain saved to %s.\n", outfile)

    return outfile


def load_extracts(chain_file):
    """Load extracted tracer number density chains.

    Parameters
    ----------
    chain_file : :class:`pathlib.Path` *or str*
        Extracted tracer number density chain file.

    Returns
    -------
    extracts : :class:`numpy.ndarray`
        Tracer number density samples.

    """
    with hp.File(PATHOUT/chain_file, 'r') as chain_data:
        extracts = chain_data['extract/chain'][()]

    return extracts


def view_extracts(chain):
    """View the extracted chain of tracer number density.

    Parameters
    ----------
    chain : :class:`numpy.ndarray`
        Chain.

    Returns
    -------
    chain_fig, contour_fig : :class:`matplotlib.figure.Figure`
        Chain and contour figures.

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
        "Saved distribution plot of tracer number density samples.\n"
    )

    return distribution_fig


# Model-independent settings.
# pylint: disable=no-member
BASE10_LOG = True
COSMOLOGY = cosmology.Planck15

# Model-specific settings.
LUMINOSITY_VARIABLE = 'magnitude'
THRESHOLD_VARIABLE = 'magnitude'
PARAMETERS = {
    'quasar_PLE': [
        'm_\\ast(z_\\mathrm{p})', '\\lg\\Phi_\\ast',
        '\\alpha_\\mathrm{l}', '\\alpha_\\mathrm{h}',
        '\\beta_\\mathrm{l}', '\\beta_\\mathrm{h}',
        'k_{1\\mathrm{l}}', 'k_{1\\mathrm{h}}',
        'k_{2\\mathrm{l}}', 'k_{2\\mathrm{h}}',
    ],
    'quasar_hybrid': [
        'm_\\ast(0)', '\\lg\\Phi_\\ast(0)',
        '\\alpha', '\\beta',
        'k_1', 'k_2',
        'c_{1\\mathrm{a}}', 'c_{1\\mathrm{b}}',
        'c_2', 'c_3',
    ],
}

# Program-specific settings.
SAVE = True
SAVEFIG = True

if __name__ == '__main__':

    progrc = initialise()

    parameters = PARAMETERS.get(progrc.model_name)

    if progrc.apparent_to_absolute:
        luminosity_threshold = progrc.threshold \
            - cosmology.Planck15.distmod(progrc.redshift).value \
            - konstante_correction(progrc.redshift)
    else:
        luminosity_threshold = progrc.threshold

    if progrc.taski == 'extract':

        input_chain, burin, reduction = read_chains()

        with Pool() as mpool:
            extracted_chain = extract_number_density(input_chain, pool=mpool)

        output_path = save_extracts()

    if progrc.taski == 'load':
        redshift_tag = "_z{:.2f}".format(progrc.redshift)
        threshold_tag = "_m{:.1f}".format(luminosity_threshold)
        prefix = "numden" + redshift_tag + threshold_tag + "_"
        output_path = PATHOUT/(prefix + progrc.chain_file)

        extracted_chain = load_extracts(progrc.chain_file)

    figures = view_extracts(extracted_chain)
