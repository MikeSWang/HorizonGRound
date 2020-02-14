r"""Luminosity function model fitting.

Examples
--------
>>> from horizonground.lumfunc_modeller import quasar_PLE_model
>>> prior_file = PATHIN/"PLE_model_prior.txt"
>>> data_file = PATHEXT/"eBOSS_QSO_LF.txt"
>>> parameter_file = PATHEXT/"PLE_model_fits.txt"
>>> likelihood = LumFuncLikelihood(quasar_PLE_model, prior_file, data_file)
>>> with open(parameter_file, 'r') as pfile:
...     parameters = tuple(
...         map(
...             lambda var_name: var_name.strip(" "),
...             pfile.readline().strip("#").strip("\n").split(",")
...         )
...     )
...     estimates = tuple(
...         map(lambda value: float(value), pfile.readline().split(","))
...     )
...     parameter_set = dict(zip(parameters, estimates))
...     for parameter in parameters:
...         if parameter.startswith("\Delta"):
...             del parameter_set[parameter]
...
>>> print(likelihood(list(parameter_set.values())))

"""
from argparse import ArgumentParser
from pprint import pprint

import corner
import emcee as mc
import matplotlib.pyplot as plt
import numpy as np
from emcee.autocorr import AutocorrError

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

    pprint(vars(parsed_args))

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

    if prog_params.task.lower().startswith("get"):
        return log_likelihood, prior_ranges, dimension
    elif prog_params.task.lower().startswith("m"):
        # Set up backend.
        output_file = (PATHOUT/prog_params.chain_file).with_suffix('.h5')
        backend = mc.backends.HDFBackend(output_file)
        backend.reset(prog_params.nwalkers, dimension)

        # Set up sampler and initial state.
        mcmc_sampler = mc.EnsembleSampler(
            prog_params.nwalkers, dimension, log_likelihood, backend=backend
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
    list of float
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
        sampler.run_mcmc(
            ini_pos, prog_params.nsteps, progress=prog_params.quiet
        )

        samples = sampler.get_chain(flat=True, thin=prog_params.thinby)

        np.save(
            (PATHOUT/prog_params.chain_file).with_suffix('.npy'),
            samples
        )

        autocorr = sampler.get_autocorr_time()

        return autocorr


def load_chains(burnin=0, reduce=1, savefig=True):
    """Load and view a chain file.

    Parameters
    ----------
    burnin : int, optional
        Number of burn-in steps to discard (default is 0).
    reduce : int, optional
        Thinning factor for reducing the chain (default is 1).
    savefig : bool, optional
        If `True` (default), save the figure.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Corner plot.
    tau : float array_like
        Auto-correlation time estimate.

    """
    mcmc_file = PATHOUT/prog_params.chain_file

    print("\nLoading chain file: {}.h5.\n".format(mcmc_file.stem))

    reader = mc.backends.HDFBackend(
        mcmc_file.with_suffix('.h5'), read_only=True
    )

    # Visualise chain.
    chain = reader.get_chain()

    chain_fig, axes = plt.subplots(ndim, figsize=(ndim, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(chain[:, ::(prog_params.nwalkers//10), i], 'k', alpha=0.25)
        ax.set_xlim(0, len(chain))
    axes[-1].set_xlabel("steps")

    if SAVEFIG:
        chain_fig.savefig(mcmc_file.with_suffix('.chain.pdf'), format='pdf')

    # Flatten the chain.
    chain_flat = reader.get_chain(
        flat=True, discard=prog_params.burnin, thin=prog_params.thinby
    )

    try:
        tau = reader.get_autocorr_time()
    except AutocorrError as ae:
        print(ae)
        tau = [np.nan] * chain_flat.shape[-1]

    contour_fig = corner.corner(chain_flat, quiet=True, rasterized=True)

    if SAVEFIG:
        contour_fig.savefig(
            mcmc_file.with_suffix('.contour.pdf'), format='pdf'
        )

    return tau


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
