"""Shift a chain of sampled parameters.

"""
from argparse import ArgumentParser
from pprint import pformat

import corner
import emcee as mc
import h5py as hp
import numpy as np

from conf import PATHIN, PATHOUT, logger
from horizonground.utils import load_parameter_set


def initialise():
    """Initialise program.

    Returns
    -------
    program_configuration : :class:`argparse.Namespace`
        Parsed program configuration parameters.

    """
    parser = ArgumentParser("shift-chain-samples")

    parser.add_argument('--original-param-file', type=str)
    parser.add_argument('--new-param-file', type=str)

    parser.add_argument('--model', type=str, default='quasar_PLE')
    parser.add_argument(
        '--sampler', type=str.lower,
        choices=['emcee', 'zeus'], default='zeus'
    )
    parser.add_argument('--chain-file', type=str, default=None)
    parser.add_argument('--burnin', type=int, default=None)
    parser.add_argument('--reduction', type=int, default=1)

    program_configuration = parser.parse_args()

    logger.info(
        "\n---Program configuration---\n%s\n",
        pformat(vars(program_configuration)).lstrip("{").rstrip("}")
    )

    return program_configuration


def read_chains():
    """Load and process chains from file.

    Returns
    -------
    chains : :class:`numpy.ndarray`
        Flattened chains.

    """
    chain_file = PATHOUT/progrc.chain_file

    if progrc.sampler == 'emcee':
        reader = mc.backends.HDFBackend(chain_file, read_only=True)
        chains = reader.get_chain()
    elif progrc.sampler == 'zeus':
        with hp.File(chain_file, 'r') as chain_data:
            chains = chain_data['mcmc']['chain'][()]

    logger.info("Loaded chain file: %s.\n", chain_file)

    return chains


def shift_chains(chains):
    """Shift parameter sample chains.

    Parameters
    ----------
    chains : :class:`numpy.ndarray`
        Parameter sample chains.

    Returns
    -------
    shifted_chains : :class:`numpy.ndarray`
        Shifted chains.

    """
    original_params = load_parameter_set(
        PATHIN/"cabinet"/progrc.original_param_file
    )
    new_params = load_parameter_set(
        PATHIN/"cabinet"/progrc.new_param_file
    )

    plus_shift = np.array([[[
        new_params[name] - original_params[name]
        for name in PARAMETERS[progrc.model]
    ]]])
    shifted_chains = chains + plus_shift

    return shifted_chains


def save_chains(shifted_chains):
    """Save shifted parameter sample chains.

    Parameters
    ----------
    shifted_chains : :class:`numpy.ndarray`
        Shifted chains.

    """
    infile = PATHOUT/progrc.chain_file
    outfile = PATHOUT/progrc.chain_file.replace(".h5", "shifted.h5")

    with hp.File(infile, 'r') as indata, hp.File(outfile, 'w') as outdata:
        outdata.create_group('mcmc')
        outdata.create_dataset('mcmc/chain', data=shifted_chains)
        outdata.create_dataset(
            'mcmc/autocorr_time', data=indata['mcmc/autocorr_time'][()]
        )

    logger.info(
        "Verify medians: {}.\n".format(
            np.squeeze([
                np.squeeze(corner.quantile(
                    shifted_param_chain.reshape(
                        -1, len(PARAMETERS[progrc.model]
                    )),
                    q=[0.5]
                ))
                for shifted_param_chain in np.transpose(shifted_chains)
            ])
        )
    )

    logger.info("Shifted chain saved to %s.\n", outfile)


PARAMETERS = {
    'quasar_PLE': [
        'm_\\ast(z_\\mathrm{p})', '\\lg\\Phi_\\ast',
        '\\alpha_\\mathrm{l}', '\\alpha_\\mathrm{h}',
        '\\beta_\\mathrm{l}', '\\beta_\\mathrm{h}',
        'k_{1\\mathrm{l}}', 'k_{1\\mathrm{h}}',
        'k_{2\\mathrm{l}}', 'k_{2\\mathrm{h}}',
    ],
    'quasar_hybrid': [
        'm_\\ast(0)', '\\lg\\Phi_\\ast(0)',
        '\\alpha', '\\beta',
        'k_1', 'k_2',
        'c_{1\\mathrm{a}}', 'c_{1\\mathrm{b}}',
        'c_2', 'c_3',
    ],
}

if __name__ == '__main__':

    progrc = initialise()

    input_chain = read_chains()
    shifted_chain = shift_chains(input_chain)
    save_chains(shifted_chain)
