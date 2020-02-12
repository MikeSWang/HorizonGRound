r"""Luminosity function model fitting.

Examples
--------
>>> parameter_file = "../data/external/PLE_model_fits.txt"
>>> likelihood = LumFuncLikelihood(quasar_PLE_model, PRIOR_FILE, DATA_FILE)
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

import emcee as mc
import numpy as np

from config import PATHEXT, PATHIN, PATHOUT, major_version, use_local_package

use_local_package("../../HorizonGRound/")

from horizonground.lumfunc_likelihood import LumFuncLikelihood


def parse_ext_args():
    """Parse external arguments.

    Returns
    -------
    :class:`argparse.Namespace`
        Parsed arguments.

    """
    parser = ArgumentParser("luminosity-function-fitting")

    parser.add_argument('--data-file', type=str, default=None)
    parser.add_argument('--prior-file', type=str, default=None)
    parser.add_argument('--chain-file', type=str, default=None)

    parser.add_argument('--nwalkers', type=int, default=100)
    parser.add_argument('--nsteps', type=int, default=10000)

    return parser.parse_args()


def setup_sampler():
    """Set up likelihood sampler.

    Returns
    -------
    sampler : :class:`emcee.EnsembleSampler`
        Likelihood sampler.
    initial_state : :class:`numpy.ndarray`
        Initial parameter-space state.
    dimension : int
        Dimension of the parameter space.

    """
    from horizonground.lumfunc_modeller import quasar_PLE_model

    log_likelihood = LumFuncLikelihood(
        quasar_PLE_model,
        PATHIN/prog_params.prior_file,
        PATHEXT/prog_params.data_file
    )

    dimension = len(log_likelihood.prior)

    sampler = mc.EnsembleSampler(
        prog_params.nwalkers, dimension, log_likelihood
    )

    prior_ranges = list(log_likelihood.prior.values())

    initial_state = \
        np.ones((prog_params.nwalkers, 1)) \
            * np.mean(prior_ranges, axis=1) \
        + np.random.randn(prog_params.nwalkers, dimension) \
            * np.diff(prior_ranges, axis=1).reshape(-1)

    return sampler, initial_state, dimension


if __name__ == '__main__':

    prog_params = parse_ext_args()

    pprint(prog_params)

    sampler, ini_pos, ndim = setup_sampler()

    if major_version(mc) >= 3:
        sampler.run_mcmc(ini_pos, prog_params.nsteps, progress=True)
        samples = sampler.get_chain(flat=True)
    else:
        sampler.run_mcmc(ini_pos, prog_params.nsteps)
        samples = sampler.chain.reshape((-1, ndim))

    np.save((PATHOUT/prog_params.chain_file).with_suffix('.npy'), samples)

    if major_version(mc) >= 3:
        autocorr = sampler.get_autocorr_time()
    else:
        autocorr = sampler.acor
    try:
        print("Auto-correlation estimate: {}. ".format(autocorr))
    except:
        print("Auto-correlation estimate unavailable. ".format(autocorr))
