r"""Luminosity function model fitting with ``zeus``.

"""
import os
from argparse import ArgumentParser
from collections import OrderedDict
from multiprocessing import Pool
from pprint import pformat

os.environ['OMP_NUM_THREADS'] = '1'

import corner
import h5py as hp
import matplotlib.pyplot as plt
import numpy as np
import zeus

from config import PATHEXT, PATHIN, PATHOUT
from config import logger
from config import sci_notation, use_local_package

use_local_package("../../HorizonGRound/")

import horizonground.lumfunc_modeller as modeller
from horizonground.lumfunc_likelihood import LumFuncLikelihood


def parse_ext_args():
    """Parse external arguments.

    Returns
    -------
    parsed_args : :class:`argparse.Namespace`
        Parsed program arguments.

    """
    parser = ArgumentParser("luminosity-function-fitting")

    parser.add_argument(
        '--task', type=str.lower,
        choices=['make', 'get', 'resume'], default='make'
    )
    parser.add_argument(
        '--mode', type=str.lower,
        choices=['continuous', 'dump'], default='dump'
    )
    parser.add_argument('--use-prior', action='store_true')
    parser.add_argument('--use-constraint', action='store_true')

    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--fixed-file', type=str, default=None)
    parser.add_argument('--prior-file', type=str, default=None)
    parser.add_argument('--chain-file', type=str, default=None)

    parser.add_argument('--nwalkers', type=int, default=100)
    parser.add_argument('--nsteps', type=int, default=10000)
    parser.add_argument('--thinby', type=int, default=1)

    parser.add_argument('--skip-chains', type=int, default=None)
    parser.add_argument('--burnin', type=int, default=None)
    parser.add_argument('--reduce', type=int, default=None)

    parsed_args = parser.parse_args()

    parsed_args.mode = parsed_args.mode \
        if parsed_args.task != 'get' else "plot"

    parsed_args.chain_file += "_{}_{}_by{}".format(
        parsed_args.nwalkers,
        sci_notation(parsed_args.nsteps),
        parsed_args.thinby
    )

    logger.info(
        "\n---Program configuration---\n%s\n", pformat(vars(parsed_args))
    )

    return parsed_args


def initialise_sampler():
    """Initialise the likelihood sampler.

    Returns
    -------
    log_likelihood : :class:`lumfunc_likelihood.LumFuncLikelihood`
        Logarithmic likelihood.
    prior_ranges : :class:`numpy.ndarray`
        Parameter-space boundaries.
    mcmc_sampler : :class:`emcee.EnsembleSampler`
        Likelihood sampler.
    initial_state : :class:`numpy.ndarray`
        Initial parameter-space state.
    dimension : int
        Dimension of the parameter space.

    """
    # Set up likelihood and prior.
    lumfunc_model = getattr(modeller, prog_params.model_name)

    lumfunc_model_constraint = getattr(
        modeller, prog_params.model_name + '_constraint', None
    ) if prog_params.use_constraint else None

    fixed_file = PATHIN/prog_params.fixed_file \
        if prog_params.fixed_file \
        else None

    log_likelihood = LumFuncLikelihood(
        lumfunc_model,
        PATHEXT/prog_params.data_file,
        PATHIN/prog_params.prior_file,
        fixed_file=fixed_file,
        model_constraint=lumfunc_model_constraint
    )

    logger.info(
        "\n---Prior parameters---\n%s\n",
        pformat(dict(log_likelihood.prior.items()))
    )
    if log_likelihood.fixed:
        logger.info(
            "\n---Fixed parameters---\n%s\n",
            pformat(dict(log_likelihood.fixed.items()))
        )

    # Set up numerics.
    dimension = len(log_likelihood.prior)
    prior_ranges = np.array(list(log_likelihood.prior.values()))

    if prog_params.task == "get":
        return log_likelihood, prior_ranges, dimension

    if prog_params.task == "make":
        pass

    # Set up sampler and initial state.
    mcmc_sampler = zeus.sampler(
        log_likelihood, prog_params.nwalkers, dimension, pool=pool,
        kwargs={'use_prior': prog_params.use_prior}
    )

    def _initialise_state():

        if lumfunc_model_constraint:
            _ini_pos = []
            while len(_ini_pos) < prog_params.nwalkers:
                criterion = False
                while not criterion:
                    pos = np.random.uniform(
                        prior_ranges[:, 0], prior_ranges[:, -1]
                    )
                    criterion = lumfunc_model_constraint(
                        dict(zip(log_likelihood.prior.keys(), pos))
                    )
                _ini_pos.append(pos)
        else:
            _ini_pos = np.random.uniform(
                prior_ranges[:, 0], prior_ranges[:, -1],
                size=(prog_params.nwalkers, dimension)
            )

        return np.asarray(_ini_pos)

    initial_state = _initialise_state()

    logger.info(
        "\n---Starting positions (walkers, parameters)---\n%s\n%s...\n",
        pformat(list(log_likelihood.prior.keys()), width=79, compact=True),
        pformat(
            np.array2string(
                initial_state[::(prog_params.nwalkers // 10), :],
                formatter={'float_kind': '{:.2f}'.format}
            )
        )
        .replace("(", "").replace(")", "")
        .replace("' ", "").replace("'", "")
        .replace("\\n", "")
    )

    return mcmc_sampler, initial_state, dimension


def run_sampler():
    """Run sampler.

    Returns
    -------
    float array_like
        Auto-correlation time estimate.

    """
    if prog_params.mode == 'continuous':
        sampler.run(ini_pos, prog_params.nsteps, progress=True)

        mcmc_file = (PATHOUT/prog_params.chain_file).with_suffix('.h5')
        with hp.File(mcmc_file, 'w') as outdata:
            outdata.create_group('mcmc')
            outdata.create_dataset('mcmc/chain', data=sampler.chain)
            outdata.create_dataset(
                'mcmc/autocorr_time', data=sampler.autocorr_time
            )
    elif prog_params.mode == 'dump':
        sampler.run(ini_pos, prog_params.nsteps, progress=True)

        mcmc_results = {
            'chain': sampler.chain,
            'autocorr_time': sampler.autocorr_time,
        }
        mcmc_file = (PATHOUT/prog_params.chain_file).with_suffix('.npy')

        np.save(mcmc_file, mcmc_results)

    return sampler.autocorr_time, sampler.efficiency, sampler.ess


def load_chains():
    """Load and view a chain file.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Corner plot.
    tau : float array_like
        Auto-correlation time estimate.

    """
    COLOUR = "#A3C1AD"
    QUANTILES = [0.1587, 0.5, 0.8413]
    levels = 1.0 - np.exp(- np.square([1, 2]) / 2)
    corner_opt = dict(
        quiet=True, rasterized=True, show_titles=True,
        plot_datapoints=False, plot_contours=True, fill_contours=True,
        quantiles=QUANTILES, levels=levels, color=COLOUR,
        truth_color='#7851a9', label_kwargs={'visible': False},
        bins=100, smooth=0.4,
    )

    # Parameter labels.
    labels = list(
        map(lambda s: "$" + s + "$", list(log_likelihood.prior.keys()))
    )
    truth = None
    if TRUTH_FILE:
        with open(TRUTH_FILE, 'r') as pfile:
            parameters = tuple(
                map(
                    lambda var_name: var_name.strip(" "),
                    pfile.readline().strip("#").strip("\n").split(",")
                )
            )
            estimates = tuple(map(float, pfile.readline().split(",")))

        model_parameters = OrderedDict(zip(parameters, estimates))
        for par_name in list(model_parameters.keys()):
            if "Delta" in par_name:
                del model_parameters[par_name]
        truth = list(model_parameters.values())

    # Load the chain.
    mcmc_file = (PATHOUT/prog_params.chain_file).with_suffix('.npy')

    mcmc_results = np.load(mcmc_file).item()

    logger.info("Loaded chain file: %s.npy.\n", mcmc_file.stem)

    # Get autocorrelation time, burn-in and thin.
    tau = mcmc_results['autocorr_time']

    if prog_params.burnin is None:
        try:
            burnin = 4 * int(np.max(tau))  # can change 4 to 2
        except ValueError:
            burnin = 0
    else:
        burnin = prog_params.burnin

    if prog_params.reduce is None:
        try:
            reduce = int(np.min(tau)) // 5  # can change 5 to 2
        except ValueError:
            reduce = 1
    else:
        reduce = prog_params.reduce

    logger.info("Burn-in set to %i. Thinning set to %i.\n", burnin, reduce)

    chains = np.swapaxes(mcmc_results['chain'], 0, 1)[burnin:, :, :]
    chain_flat = chains[::reduce, :, :].reshape((-1, ndim))

    # Visualise chain.
    plt.close('all')

    chains_fig, axes = plt.subplots(ndim, figsize=(12, ndim), sharex=True)

    skip_chains = 1 if prog_params.skip_chains is None \
        else prog_params.nwalkers // prog_params.skip_chains
    for param_idx in range(ndim):
        ax = axes[param_idx]
        ax.plot(
            chains[:, ::skip_chains, param_idx],
            alpha=0.66, rasterized=True
        )
        ax.set_xlim(0, len(chains))
        ax.set_ylabel(labels[param_idx])
    axes[-1].set_xlabel("steps")

    if SAVEFIG:
        chains_fig.savefig(mcmc_file.with_suffix('.chains.pdf'), format='pdf')
    logger.info("Saved plot of chains.\n")

    chain_flat_fig, axes = plt.subplots(ndim, figsize=(12, ndim), sharex=True)
    for param_idx in range(ndim):
        ax = axes[param_idx]
        ax.plot(
            chain_flat[:, param_idx], color=COLOUR, alpha=0.66, rasterized=True
        )
        ax.set_xlim(0, len(chain_flat))
        ax.set_ylabel(labels[param_idx])
    axes[-1].set_xlabel("steps")

    if SAVEFIG:
        chain_flat_fig.savefig(
            mcmc_file.with_suffix('.flatchain.pdf'), format='pdf'
        )
    logger.info("Saved plot of flattened chains.\n")

    contour_fig = corner.corner(
        chain_flat, labels=labels, truths=truth, **corner_opt
    )

    if SAVEFIG:
        contour_fig.savefig(mcmc_file.with_suffix('.pdf'), format='pdf')
    logger.info("Saved contour plot.\n")

    return tau


if __name__ == '__main__':

    SAVEFIG = True
    TRUTH_FILE = "../data/external/PLE_model_fits.txt"

    prog_params = parse_ext_args()

    if prog_params.task == 'make':
        with Pool() as pool:
            sampler, ini_pos, ndim = initialise_sampler()
            autocorr_est, efficiency, effective_size = run_sampler()

        logger.info(
            "Efficiency %.2f; effective size %i.\n", efficiency, effective_size
        )
    elif prog_params.task == 'get':
        log_likelihood, prior_ranges, ndim = initialise_sampler()
        autocorr_est = load_chains()
    logger.info(
        "Auto-correlation time estimate: %s.\n",
        ["{:.0f}".format(act) for act in autocorr_est]
    )
