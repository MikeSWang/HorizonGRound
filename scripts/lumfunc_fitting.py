r"""Luminosity function model fitting.

Examples
--------
>>> from horizonground.lumfunc_modeller import quasar_PLE_model
>>> prior_file = PATHIN/"PLE_model_prior.txt"
>>> data_file = PATHEXT/"eBOSS_QSO_LF.txt"
>>> parameter_file = PATHEXT/"PLE_model_invalid.txt"
>>> likelihood = LumFuncLikelihood(quasar_PLE_model, prior_file, data_file)
>>> parameter_set = load_parameter_fits(parameter_fits_file)
>>> print(likelihood(list(parameter_set.values()), use_prior=True))

"""
import os
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Pool
from pprint import pprint

os.environ["OMP_NUM_THREADS"] = "1"

import corner
import emcee as mc
import matplotlib.pyplot as plt
import numpy as np
from emcee.autocorr import AutocorrError

from config import PATHEXT, PATHIN, PATHOUT, sci_notation, use_local_package

use_local_package("../../HorizonGRound/")

import horizonground.lumfunc_modeller as lumfunc_modeller
from horizonground.lumfunc_likelihood import LumFuncLikelihood


def load_parameter_fits(parameter_fits_file):

    with open(parameter_fits_file, 'r') as pfile:
        parameters = tuple(
            map(
                lambda var_name: var_name.strip(" "),
                pfile.readline().strip("#").strip("\n").split(",")
            )
        )
        estimates = tuple(
            map(lambda value: float(value), pfile.readline().split(","))
        )
        parameter_set = dict(zip(parameters, estimates))
        for parameter in parameters:
            if parameter.startswith("\Delta"):
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
    parser.add_argument('--use-prior', action='store_true')

    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--prior-file', type=str, default=None)
    parser.add_argument('--fixed-file', type=str, default=None)
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

    fixed_file = PATHIN/prog_params.fixed_file \
        if prog_params.fixed_file \
        else None

    log_likelihood = LumFuncLikelihood(
        lumfunc_model,
        PATHIN/prog_params.prior_file,
        PATHEXT/prog_params.data_file,
        fixed_file=fixed_file
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
        backend=backend, pool=pool
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
    mcmc_file = PATHOUT/prog_params.chain_file

    print("\nLoading chain file: {}.h5.\n".format(mcmc_file.stem))

    reader = mc.backends.HDFBackend(
        mcmc_file.with_suffix('.h5'), read_only=True
    )

    try:
        tau = reader.get_autocorr_time()
    except AutocorrError as ae:
        print("\n", ae, "\n")
        tau = [np.nan] * len(labels)

    if prog_params.burnin == 0:
        try:
            burnin = 2 * int(np.max(tau))
        except ValueError:
            burnin = prog_params.burnin
    else:
        burnin = prog_params.burnin
    if prog_params.reduce == 1:
        try:
            reduce = int(np.min(tau)) // 2
        except ValueError:
            reduce = prog_params.reduce
    else:
        reduce = prog_params.reduce

    chain = reader.get_chain(flat=True, discard=burnin, thin=reduce)

    # Visualise chain.
    plt.close('all')

    chain_fig, axes = plt.subplots(ndim, figsize=(ndim, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(chain[:, i], color=COLOUR, alpha=0.66, rasterized=True)
        ax.set_xlim(0, len(chain))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("steps")

    if SAVEFIG:
        chain_fig.savefig(mcmc_file.with_suffix('.chain.pdf'), format='pdf')

    contour_fig = corner.corner(chain, labels=labels, **corner_opt)
    if SAVEFIG:
        contour_fig.savefig(
            mcmc_file.with_suffix('.contour.pdf'), format='pdf'
        )

    return tau


if __name__ != '__main__':

    SAVEFIG = True

    prog_params = parse_ext_args()

    if prog_params.task in ['make', 'resume']:
        with Pool() as pool:
            sampler, ini_pos, ndim = initialise_sampler()
            autocorr_est = run_sampler()
    elif prog_params.task == 'get':
        log_likelihood, prior_ranges, ndim = initialise_sampler()
        autocorr_est = load_chains()

    print("\nAuto-correlation time estimate: {}.\n"
      .format(["{:.2f}".format(act) for act in autocorr_est])
    )
