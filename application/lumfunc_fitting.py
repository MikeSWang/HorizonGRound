r"""Luminosity function model fitting.

Examples
--------
>>> from horizonground.lumfunc_modeller import quasar_PLE_lumfunc
>>> measurement_file = PATHEXT/"eBOSS_QSO_LF_measurements.txt"
>>> uncertainty_file = PATHEXT/"eBOSS_QSO_LF_uncertainties.txt"
>>> prior_file = PATHIN/"QSO_LF_PLE_model_prior.txt"
>>> likelihood = LumFuncLikelihood(
...     quasar_PLE_lumfunc, measurement_file, prior_file,
...     uncertainty_file=uncertainty_file, prescription='native'
... )
>>> parameter_set_file = PATHIN/"cabinet"/"QSO_LF_PLE_model_parameters.txt"
>>> parameter_set = load_parameter_set(parameter_set_file)
>>> print(likelihood(list(parameter_set.values()), use_prior=True))

"""
from argparse import ArgumentParser
from multiprocessing import Pool
from pprint import pformat

import corner
import emcee as mc
import h5py as hp
import matplotlib.pyplot as plt
import numpy as np
import zeus
from emcee.autocorr import AutocorrError

from conf import PATHEXT, PATHIN, PATHOUT, logger, sci_notation
from horizonground.lumfunc_likelihood import LumFuncLikelihood
from horizonground.utils import load_parameter_set
import horizonground.lumfunc_modeller as modeller


def parse_ext_args():
    """Parse external arguments.

    Returns
    -------
    parsed_args : :class:`argparse.Namespace`
        Parsed program arguments.

    """
    parser = ArgumentParser("luminosity-function-fitting")

    parser.add_argument(
        '--task', type=str.lower, choices=['sample', 'retrieve', 'resume'],
        help="task to perform: sample; retrieve samples; or resume sampling"
    )
    parser.add_argument('--sampler', type=str.lower, choices=['emcee', 'zeus'])
    parser.add_argument(
        '--prescription', type=str.lower,
        choices=['native', 'poisson', 'symlg']
    )
    parser.add_argument(
        '--jump', action='store_true',
        help="change zeus sampler jump condition; used for multimodal cases"
    )

    parser.add_argument('--use-prior', action='store_true')
    parser.add_argument('--use-constraint', action='store_true')
    parser.add_argument(
        '--nonautostop', action='store_true',
        help="do not stop sampling after convergence detected"
    )
    parser.add_argument('--quiet', action='store_false')

    parser.add_argument('--model-name', type=str, default='quasar_PLE')
    parser.add_argument('--data-files', type=str, nargs=2, default=None)
    parser.add_argument('--prior-file', type=str, default=None)
    parser.add_argument('--fixed-file', type=str, default=None)
    parser.add_argument('--chain-file', type=str, default=None)

    parser.add_argument('--nwalkers', type=int, default=100)
    parser.add_argument('--nsteps', type=int, default=10000)
    parser.add_argument('--thinby', type=int, default=1)

    parser.add_argument(
        '--skip-chains', type=int, default=None,
        help="only plot chain traces every interval"
    )
    parser.add_argument('--burnin', type=int, default=None)
    parser.add_argument('--reduce', type=int, default=None)

    parsed_args = parser.parse_args()

    parsed_args.chain_file = parsed_args.chain_file.rstrip('.h5')
    parsed_args.chain_file += "_{}_{}_{}_by{}.h5".format(
        parsed_args.prescription,
        parsed_args.nwalkers,
        sci_notation(parsed_args.nsteps),
        parsed_args.thinby
    )

    logger.info(
        "\n---Program configuration---\n%s\n",
        pformat(vars(parsed_args)).lstrip("{").rstrip("}")
    )

    return parsed_args


def initialise_sampler():
    """Initialise the likelihood sampler.

    Returns
    -------
    likelihood : :class:`lumfunc_likelihood.LumFuncLikelihood`
        Logarithmic likelihood.
    prior_ranges : :class:`numpy.ndarray`
        Parameter-space boundaries.
    mcmc_sampler : :class:`emcee.EnsembleSampler` or :class:`zeus.sampler`
        Markov chain Monte Carlo sampler.
    initial_state : :class:`numpy.ndarray`
        Initial parameter-space state.
    dimension : int
        Dimension of the parameter space.

    """
    # Set up likelihood and prior.
    lumfunc_model = getattr(modeller, prog_params.model_name + '_lumfunc')

    measurement_file, uncertainty_file = prog_params.data_files

    fixed_file = PATHIN/prog_params.fixed_file \
        if prog_params.fixed_file else None

    model_constraint = getattr(
        modeller, prog_params.model_name + '_constraint', None
    ) if prog_params.use_constraint else None

    likelihood = LumFuncLikelihood(
        lumfunc_model,
        PATHEXT/measurement_file,
        PATHIN/prog_params.prior_file,
        uncertainty_file=PATHEXT/uncertainty_file,
        fixed_file=fixed_file,
        prescription=prog_params.prescription,
        model_constraint=model_constraint
    )

    logger.info(
        "\n---Prior parameters---\n%s\n",
        pformat(dict(likelihood.prior.items()))
    )
    if likelihood.fixed:
        logger.info(
            "\n---Fixed parameters---\n%s\n",
            pformat(dict(likelihood.fixed.items()))
        )

    dimension = len(likelihood.prior)
    _prior_ranges = np.array(list(likelihood.prior.values()))

    if prog_params.task == 'retrieve':
        return likelihood, _prior_ranges, dimension

    # Set up the sampler.
    if prog_params.sampler == 'emcee':
        output_file = (PATHOUT/prog_params.chain_file).with_suffix('.h5')

        backend = mc.backends.HDFBackend(output_file)
        if prog_params.task == 'sample':
            backend.reset(prog_params.nwalkers, dimension)

        mcmc_sampler = mc.EnsembleSampler(
            prog_params.nwalkers, dimension, likelihood,
            kwargs={'use_prior': prog_params.use_prior},
            backend=backend, pool=pool
        )
    elif prog_params.sampler == 'zeus':
        if prog_params.jump:
            proposal = {
                'differential': 0.85,
                'gaussian': 0.0,
                'jump': 0.15,
                'random': 0.0,
            }
        else:
            proposal = {
                'differential': 1.0,
                'gaussian': 0.0,
                'jump': 0.0,
                'random': 0.0,
            }

        mcmc_sampler = zeus.sampler(
            likelihood, prog_params.nwalkers, dimension,
            proposal=proposal, pool=pool,
            kwargs={'use_prior': prog_params.use_prior}
        )

    # Set up the initial state.
    def _initialise_state():
        if model_constraint:
            _ini_pos = []
            while len(_ini_pos) < prog_params.nwalkers:
                criterion = False
                while not criterion:
                    pos = np.random.uniform(
                        _prior_ranges[:, 0], _prior_ranges[:, -1]
                    )
                    criterion = model_constraint(
                        dict(zip(likelihood.prior.keys(), pos))
                    )
                _ini_pos.append(pos)
        else:
            _ini_pos = np.random.uniform(
                _prior_ranges[:, 0], _prior_ranges[:, -1],
                size=(prog_params.nwalkers, dimension)
            )
        return np.asarray(_ini_pos)

    if prog_params.task == "resume":
        initial_state = None
    else:
        initial_state = _initialise_state()

        logger.info(
            "\n---Starting positions (walkers, parameters)---\n%s\n%s...\n",
            pformat(list(likelihood.prior.keys()), width=79, compact=True)
            .lstrip("{").rstrip("}"),
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
    list of float
        Auto-correlation time estimate.

    """
    KNOT_LENGTH = 100
    CONVERGENCE_TOL = 0.01

    # Use ``emcee`` sampler.
    if prog_params.sampler == 'emcee':

        if prog_params.task == 'sample':

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

        if prog_params.task == 'resume':

            sampler.run_mcmc(
                ini_pos, prog_params.nsteps,
                thin_by=prog_params.thinby,
                progress=prog_params.quiet
            )

            autocorr_estimate = sampler.get_autocorr_time()

            return autocorr_estimate[-1]

    # Use ``zeus`` sampler.
    if prog_params.sampler == 'zeus':

        sampler.run(ini_pos, prog_params.nsteps, progress=True)

        output_file = PATHOUT/prog_params.chain_file
        with hp.File(output_file, 'w') as outdata:
            outdata.create_group('mcmc')
            outdata.create_dataset(
                'mcmc/chain', data=np.swapaxes(sampler.chain, 0, 1)
            )
            outdata.create_dataset(
                'mcmc/autocorr_time', data=sampler.autocorr_time
            )
            outdata.create_dataset(
                'mcmc/effective_sample_size', data=sampler.ess
            )
            outdata.create_dataset('mcmc/efficiency', data=sampler.efficiency)

        return sampler.autocorr_time

    raise ValueError(f"Unknown MCMC sampler {prog_params.sampler}.")


def load_chains():
    """Load and view a chain file.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        Corner plot.
    tau : float array_like
        Auto-correlation time estimate.

    """
    quantiles = [0.1587, 0.5, 0.8413]
    levels = 1.0 - np.exp(- np.square([1, 2]) / 2)
    corner_opt = dict(
        quiet=True, rasterized=True,
        show_titles=True, label_kwargs={'visible': False},
        plot_datapoints=False, plot_contours=True, fill_contours=True,
        quantiles=quantiles, levels=levels,
        color='#a3c1ad', truth_color='#7851a9',
        bins=160, smooth=0.45, range=(0.999,)*ndim
    )

    # Parameter labels.
    labels = list(
        map(lambda s: "$" + s + "$", list(log_likelihood.prior.keys()))
    )
    truth = None
    if truth_file:
        external_fits = load_parameter_set(truth_file)
        truth = list(external_fits.values())

    # Load the chain.
    mcmc_file = PATHOUT/prog_params.chain_file

    if prog_params.sampler == 'emcee':
        reader = mc.backends.HDFBackend(mcmc_file, read_only=True)
    elif prog_params.sampler == 'zeus':
        mcmc_results = hp.File(mcmc_file, 'r')
        reader = mcmc_results['mcmc']

    logger.info("Loaded chain file: %s.\n", mcmc_file.name)

    # Get autocorrelation time, burn-in and thin.
    try:
        tau = reader.get_autocorr_time()
    except AttributeError:
        tau = reader['autocorr_time'][()]
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

    if prog_params.sampler == 'emcee':
        chains = reader.get_chain(discard=burnin, thin=reduce)
        chain_flat = reader.get_chain(flat=True, discard=burnin, thin=reduce)
    elif prog_params.sampler == 'zeus':
        chains = reader['chain'][burnin::reduce, :, :]
        chain_flat = chains.reshape((-1, ndim))
        mcmc_results.close()

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

    fig_file = str(mcmc_file).replace('.h5', '.chains.pdf')
    if SAVEFIG:
        chains_fig.savefig(fig_file)
    logger.info("Saved plot of chains.\n")

    contour_fig = corner.corner(
        chain_flat, labels=labels, truths=truth, **corner_opt
    )

    fig_file = str(mcmc_file).replace('.h5', '.contours.pdf')
    if SAVEFIG:
        contour_fig.savefig(fig_file)
    logger.info("Saved contour plot.\n")

    return tau


SAVEFIG = True
TRUTH_FILE = {
    'quasar_PLE': PATHEXT/"eBOSS_QSO_LF_PLE_model_fits.txt",
    'quasar_hybrid': PATHEXT/"eBOSS_QSO_LF_PLE+LEDE_model_fits.txt",
}

if __name__ == '__main__':

    prog_params = parse_ext_args()

    truth_file = TRUTH_FILE.get(prog_params.model_name)

    if prog_params.task in ['sample', 'resume']:
        with Pool() as pool:
            sampler, ini_pos, ndim = initialise_sampler()
            autocorr_est = run_sampler()
    elif prog_params.task == 'retrieve':
        log_likelihood, prior_ranges, ndim = initialise_sampler()
        autocorr_est = load_chains()

    logger.info(
        "Auto-correlation time estimate: %s.\n",
        ["{:.0f}".format(act) for act in autocorr_est]
    )
