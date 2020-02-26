r"""Luminosity function model fitting.

Examples
--------
>>> from horizonground.lumfunc_modeller import quasar_PLE_model
>>> data_file = PATHEXT/"eBOSS_QSO_LF.txt"
>>> prior_file = PATHIN/"PLE_model_prior_varied.txt"
>>> fixed_file = PATHIN/"PLE_model_prior_fixed.txt"
>>> likelihood = LumFuncLikelihood(
...     quasar_PLE_model, data_file, prior_file, fixed_file=fixed_file
... )
>>> parameter_set_file = PATHEXT/"PLE_model_fits.txt"
>>> parameter_set = load_parameter_set(parameter_set_file)
>>> print(likelihood(list(parameter_set.values()), use_prior=True))

"""
import os
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Pool
from pprint import pformat

os.environ['OMP_NUM_THREADS'] = '1'

import corner
import emcee as mc
import matplotlib.pyplot as plt
import numpy as np
from emcee.autocorr import AutocorrError

from config import PATHEXT, PATHIN, PATHOUT
from config import logger
from config import sci_notation, use_local_package

use_local_package("../../HorizonGRound/")

import horizonground.lumfunc_modeller as modeller
from horizonground.lumfunc_likelihood import LumFuncLikelihood


def load_parameter_set(parameter_set_file):
    """Load a parameter set from a file into a dictionary.

    Parameters
    ----------
    parameter_set_file : str or :class:`pathlib.Path`
        Parameter set file.

    Returns
    -------
    parameter_set : dict
        Parameter set.

    """
    with open(parameter_set_file, 'r') as pfile:
        parameters = tuple(
            map(
                lambda var_name: var_name.strip(" "),
                pfile.readline().strip("#").strip("\n").split(",")
            )
        )
        estimates = tuple(map(float, pfile.readline().split(",")))

    parameter_set = dict(zip(parameters, estimates))
    for parameter in parameters:
        if parameter.startswith(r"\Delta"):
            del parameter_set[parameter]

    return parameter_set


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
        choices=['continuous', 'dump'], default='continuous'
    )
    parser.add_argument('--quiet', action='store_false')
    parser.add_argument('--nonautostop', action='store_true')
    parser.add_argument('--use-prior', action='store_true')
    parser.add_argument('--use-constraint', action='store_true')

    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--prior-file', type=str, default=None)
    parser.add_argument('--fixed-file', type=str, default=None)
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

    # Set up backend.
    output_file = (PATHOUT/prog_params.chain_file).with_suffix('.h5')
    if prog_params.task == "make":
        backend = mc.backends.HDFBackend(output_file)
        backend.reset(prog_params.nwalkers, dimension)
    elif prog_params.task == "resume":
        backend = mc.backends.HDFBackend(output_file, name=str(datetime.now()))

    # Set up sampler and initial state.
    mcmc_sampler = mc.EnsembleSampler(
        prog_params.nwalkers, dimension, log_likelihood,
        kwargs={'use_prior': prog_params.use_prior},
        backend=backend, pool=pool
    )

    initial_state = np.random.uniform(
        low=prior_ranges[:, 0], high=prior_ranges[:, -1],
        size=(prog_params.nwalkers, dimension)
    )

    logger.info(
        "\n---Starting positions (~10 walkers, parameters)---\n%s...\n",
        pformat(
            np.array2string(
                initial_state[::(prog_params.nwalkers // 10), :],
                precision=2
            )
            .strip("(").strip(")")
            .replace("' ", "").replace("'", "").replace("\\n", "")
        )
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
        first_convergence_point = True
        for _ in sampler.sample(
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
                if first_convergence_point:
                    logger.info(
                        "Chain converged after %i samples.\n",
                        prog_params.nwalkers * step
                    )
                    first_convergence_point = False
                if not prog_params.nonautostop:
                    return current_tau

        return autocorr_estimate[-1]

    if prog_params.mode.startswith('dump'):
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
        quantiles=QUANTILES, color=COLOUR, levels=levels,
        label_kwargs={'visible': False}, bins=100, smooth=0.4,
    )

    # Parameter labels.
    labels = list(
        map(lambda s: "$" + s + "$", list(log_likelihood.prior.keys()))
    )

    # Load the chain.
    mcmc_file = PATHOUT/prog_params.chain_file

    reader = mc.backends.HDFBackend(
        mcmc_file.with_suffix('.h5'), read_only=True
    )

    logger.info("Loaded chain file: %s.h5.\n", mcmc_file.stem)

    # Get autocorrelation time, burn-in and thin.
    try:
        tau = reader.get_autocorr_time()
    except AutocorrError as act_warning:
        logger.warning(act_warning)
        tau = [np.nan] * len(labels)

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

    chains = reader.get_chain(discard=burnin, thin=reduce)
    chain_flat = reader.get_chain(flat=True, discard=burnin, thin=reduce)

    # Visualise chain.
    plt.close('all')

    chains_fig, axes = plt.subplots(ndim, figsize=(12, ndim), sharex=True)

    skip_chains = 1 if prog_params.skip_chains is None \
        else prog_params.nwalkers // prog_params.skip_chains
    for param_idx in range(ndim):
        ax = axes[param_idx]
        ax.plot(
            chains[:, ::skip_chains, param_idx], alpha=0.66, rasterized=True
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

    contour_fig = corner.corner(chain_flat, labels=labels, **corner_opt)

    if SAVEFIG:
        contour_fig.savefig(mcmc_file.with_suffix('.pdf'), format='pdf')
    logger.info("Saved contour plot.\n")

    return tau


if __name__ == '__main__':

    SAVEFIG = True

    prog_params = parse_ext_args()

    if prog_params.task in ['make', 'resume']:
        with Pool() as pool:
            sampler, ini_pos, ndim = initialise_sampler()
            autocorr_est = run_sampler()
    elif prog_params.task == 'get':
        log_likelihood, prior_ranges, ndim = initialise_sampler()
        autocorr_est = load_chains()

    logger.info(
        "Auto-correlation time estimate: %s",
        ["{:.1f}".format(act) for act in autocorr_est]
    )
