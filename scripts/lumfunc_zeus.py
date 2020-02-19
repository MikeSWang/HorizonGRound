r"""Luminosity function model fitting with ``zeus``.

"""
import os
from argparse import ArgumentParser
from multiprocessing import Pool
from pprint import pprint

os.environ["OMP_NUM_THREADS"] = "1"

import corner
import matplotlib.pyplot as plt
import numpy as np
import zeus

from config import PATHEXT, PATHIN, PATHOUT, sci_notation, use_local_package

use_local_package("../../HorizonGRound/")

import horizonground.lumfunc_modeller as lumfunc_modeller
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
    parser.add_argument('--quiet', action='store_false')
    parser.add_argument('--use-prior', action='store_true')

    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--fixed-file', type=str, default=None)
    parser.add_argument('--prior-file', type=str, default=None)
    parser.add_argument('--chain-file', type=str, default=None)

    parser.add_argument('--nwalkers', type=int, default=100)
    parser.add_argument('--nsteps', type=int, default=10000)
    parser.add_argument('--thinby', type=int, default=1)

    parser.add_argument('--burnin', type=int, default=0)
    parser.add_argument('--reduce', type=int, default=1)

    parsed_args = parser.parse_args()

    parsed_args.chain_file += "_{}_{}_by{}".format(
        parsed_args.nwalkers,
        sci_notation(parsed_args.nsteps),
        parsed_args.thinby
    )

    print("\nProgram parameters: ")
    pprint(vars(parsed_args))
    print("\n")

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
    lumfunc_model = getattr(lumfunc_modeller, prog_params.model_name)

    log_likelihood = LumFuncLikelihood(
        lumfunc_model,
        PATHIN/prog_params.prior_file,
        PATHEXT/prog_params.data_file,
        fixed_file=PATHIN/prog_params.fixed_file
    )

    print("\nPrior parameters: ")
    pprint(log_likelihood.prior)
    print("\n")
    print("\nFixed parameters: ")
    pprint(log_likelihood.fixed)
    print("\n")

    # Set up numerics.
    dimension = len(log_likelihood.prior)
    prior_ranges = np.array(list(log_likelihood.prior.values()))

    if prog_params.task == "get":
        return log_likelihood, prior_ranges, dimension
    elif prog_params.task == "make":
        # Set up sampler and initial state.
        mcmc_sampler = zeus.sampler(
            log_likelihood, prog_params.nwalkers, dimension, pool=pool
        )

        initial_state = np.mean(prior_ranges, axis=1) \
            + np.random.uniform(
                low=prior_ranges[:, 0], high=prior_ranges[:, -1],
                size=(prog_params.nwalkers, dimension)
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
        pass
    elif prog_params.mode.startswith('dump'):
        sampler.run(ini_pos, prog_params.nsteps, progress=True)

        mcmc_results = {
            'chain': sampler.chain,
            'autocorr_time': sampler.autocorr_time,
        }
        mcmc_file = (PATHOUT/prog_params.chain_file).with_suffix('.npy')

        np.save(mcmc_file, mcmc_results)

        return sampler.autocorr_time


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
        quiet=True, rasterized=True, show_titles=True, use_math_text=True,
        plot_datapoints=False, plot_contours=True, fill_contours=True,
        quantiles=QUANTILES, color=COLOUR,
        levels=levels, label_kwargs={'visible': False},
    )

    # Parameter labels.
    labels = list(
        map(lambda s: "$" + s + "$", list(log_likelihood.prior.keys()))
    )

    # Load the chain.
    mcmc_file = (PATHOUT/prog_params.chain_file).with_suffix('.npy')

    print("\nLoading chain file: {}.npy.\n".format(mcmc_file.stem))

    mcmc_results = np.load(mcmc_file).item()

    tau = mcmc_results['autocorr_time']

    if prog_params.burnin == 0:
        burnin = 2 * int(np.max(tau))
    else:
        burnin = prog_params.burnin
    if prog_params.reduce == 1:
        reduce = int(np.min(tau)) // 2
    else:
        reduce = prog_params.reduce

    chain = mcmc_results['chain']
    chain_flat = chain[:, burnin::reduce, :].reshape((-1, ndim), order='F')

    # Visualise chain.
    plt.close('all')

    chain_fig, axes = plt.subplots(ndim, figsize=(ndim, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(
            chain[:, ::(prog_params.nwalkers//10), i], 
            color=COLOUR, alpha=0.66, rasterized=True
        )
        ax.set_xlim(0, len(chain))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("steps")

    if SAVEFIG:
        chain_fig.savefig(mcmc_file.with_suffix('.chain.pdf'), format='pdf')

    contour_fig = corner.corner(chain_flat, labels=labels, **corner_opt)
    if SAVEFIG:
        contour_fig.savefig(
            mcmc_file.with_suffix('.contour.pdf'), format='pdf'
        )

    return tau


if __name__ == '__main__':

    SAVEFIG = True

    prog_params = parse_ext_args()

    if prog_params.task == 'make':
        with Pool() as pool:
            sampler, ini_pos, ndim = initialise_sampler()
            autocorr = run_sampler()
    elif prog_params.task == 'get':
        log_likelihood, prior_ranges, ndim = initialise_sampler()
        autocorr_est = load_chains()

    print("\nAuto-correlation estimate: {}.\n"
      .format(["{:.2f}".format(act) for act in autocorr_est])
    )
