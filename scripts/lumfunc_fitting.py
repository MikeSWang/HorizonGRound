"""Luminosity function model fitting.

"""
import numpy as np
import emcee as mc

from config import use_local_package

use_local_package("../../HorizonGRound/")

from horizonground.lumfunc_likelihood import LFLikelihood
from horizonground.lumfunc_modeller import quasar_PLE_model


DATA_FILE = "../data/input/eBOSS_QSO_LF.txt"
PRIOR_FILE = "../data/input/PLE_model_prior.txt"
PARAMETER_FILE = "../data/input/PLE_model_fits.txt"

if __name__ == '__main__':

    likelihood = LFLikelihood(quasar_PLE_model, PRIOR_FILE, DATA_FILE)

    with open(PARAMETER_FILE, 'r') as pfile:
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

    logp = likelihood(**parameter_set)
