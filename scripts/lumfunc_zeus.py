r"""Luminosity function model fitting with ``zeus``.

"""
from argparse import ArgumentParser
from pprint import pprint

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

    parser.add_argument('--task', type=str.lower, choices=['make', 'get'])
    parser.add_argument(
        '--mode', type=str.lower,
        choices=['continuous', 'dump'], default='continuous'
    )
    parser.add_argument('--quiet', action='store_false')
    parser.add_argument('--use-prior', action='store_true')

    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--prior-file', type=str, default=None)
    parser.add_argument('--chain-file', type=str, default=None)

    parser.add_argument('--nwalkers', type=int, default=100)
    parser.add_argument('--nsteps', type=int, default=10000)
    parser.add_argument('--burnin', type=int, default=0)
    parser.add_argument('--thinby', type=int, default=1)

    parsed_args = parser.parse_args()

    parsed_args.chain_file += "_{}_{}_by{}".format(
        parsed_args.nwalkers,
        sci_notation(parsed_args.nsteps),
        parsed_args.thinby
    )

    print('\n')
    pprint(vars(parsed_args))
    print('\n')

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
        PATHEXT/prog_params.data_file
    )

    # Set up numerics.
    dimension = len(log_likelihood.prior)
    prior_ranges = np.array(list(log_likelihood.prior.values()))

    if prog_params.task == "get":
        return log_likelihood, prior_ranges, dimension
    elif prog_params.task == "make":
        # Set up sampler and initial state.
        mcmc_sampler = zeus.sampler(
            log_likelihood, prog_params.nwalkers, dimension
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
    KNOT_LENGTH = 100
    CONVERGENCE_TOL = 0.01

    if prog_params.mode == 'continuous':
        autocorr_estimate = []
        step = 0
        current_tau = np.inf
        for sample in sampler.sample(
                ini_pos,
                iterations=prog_params.nsteps,
                thin_by=prog_params.thinby,
                progress=prog_params.quiet
            ):
            # Record at knot points.
            if sampler.iteration % KNOT_LENGTH:
                continue

            tau = sampler.get_autocorr_time(tol=0)

            autocorr_estimate.append(tau)
            step += 1

            # Break at convergence.
            converged = np.all(
                KNOT_LENGTH * tau < sampler.iteration
            ) & np.all(
                np.abs(tau - current_tau) < CONVERGENCE_TOL * current_tau
            )

            current_tau = tau

            if converged:
                return current_tau

        return autocorr_estimate[-1]

    elif prog_params.mode.startswith('dump'):
        sampler.run(ini_pos, prog_params.nsteps, progress=True)

        np.save(
            (PATHOUT/prog_params.chain_file).with_suffix('.npy'),
            sampler.chain
        )

        autocorr = sampler.autocorr_time

        return autocorr


def load_chains():
    """Load and view a chain file.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Corner plot.
    tau : float array_like
        Auto-correlation time estimate.

    """
    pass


if __name__ == '__main__':

    SAVEFIG = True

    prog_params = parse_ext_args()

    if prog_params.task == 'make':
        sampler, ini_pos, ndim = initialise_sampler()
        autocorr = run_sampler()
    elif prog_params.task == 'get':
        log_likelihood, prior_ranges, ndim = initialise_sampler()
        autocorr = load_chains()

    print("Auto-correlation estimate: {}. ".format(autocorr))
