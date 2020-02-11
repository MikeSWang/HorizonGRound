"""Luminosity function model fitting.

Point testing:

Examples
--------
>>> likelihood = LumFuncLikelihood(quasar_PLE_model, PRIOR_FILE, DATA_FILE)
>>> with open(PARAMETER_FILE, 'r') as pfile:
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

import numpy as np
import emcee as mc

from config import use_local_package

use_local_package("../../HorizonGRound/")

from horizonground.lumfunc_likelihood import LumFuncLikelihood


if __name__ == '__main__':

    data_file = "../data/external/eBOSS_QSO_LF.txt"
    parameter_file = "../data/external/PLE_model_fits.txt"
    prior_file = "../data/input/PLE_model_prior.txt"

    from horizonground.lumfunc_modeller import quasar_PLE_model

    log_likelihood = LumFuncLikelihood(quasar_PLE_model, prior_file, data_file)

    nwalkers = 80
    ndim = len(log_likelihood.prior)

    pos = np.mean(list(log_likelihood.prior.values()), axis=1) \
            + np.random.randn(nwalkers, ndim)

    sampler = mc.EnsembleSampler(nwalkers, ndim, log_likelihood)
    sampler.run_mcmc(pos, 50000, progress=True)
    samples = sampler.get_chain(flat=True)
    np.save("../data/output/QSO_LF_chains.npy", samples)
