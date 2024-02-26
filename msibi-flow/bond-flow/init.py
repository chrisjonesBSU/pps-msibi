#!/usr/bin/env python
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories.
The result of running this file is the creation of a signac workspace:
    - signac.rc file containing the project name
    - signac_statepoints.json summary for the entire workspace
    - workspace/ directory that contains a sub-directory of every individual statepoint
    - signac_statepoints.json within each individual statepoint sub-directory.
"""

import logging
from collections import OrderedDict
from itertools import product

import signac


def get_parameters(ordered_dict=OrderedDict()):
    '''Use the listed parameters below to set up
    your MSIBI instructions.

    '''
    parameters = ordered_dict

    # Optimizer parameters
    parameters["nlist"] = ["Cell"]
    parameters["integrator"] = ["ConstantVolume"]
    parameters["thermostat_tau"] = [0.03]
    parameters["dt"] = [0.0003]
    parameters["r_cut"] = [2.5]
    parameters["nlist_exclusions"] = [["bond", "angle"]]
    parameters["n_steps"] = [1e5]
    parameters["n_iterations"] = [10]

    # State parameters
    parameters["single_chain_path"] = [
        "/home/erjank_project/PPS-MSIBI/pps-msibi/training-runs/single-chains"]
    parameters["states"] = [
        [
            {"name": "A",
             "kT": 7.0,
             "remove_hydrogens": True,
             "alpha": 0.6,
             "n_frames": 100
             },
        ],

    ]

    # Bond parameters
    parameters["head_correction"] = ["linear"]
    parameters["bonds_nbins"] = [100]
    parameters["bonds"] = [
        [
            {"type1": "A",
             "type2": "A",
             "x0": 1.5,
             "x_min": 0,
             "x_max": 4.0,
             "k4": 0,
             "k3": 0,
             "k2": 100
             },
        ]
    ]

    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project()  # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create the generate jobs
    for params in param_combinations:
        statepoint = dict(zip(param_names, params))
        job = project.open_job(statepoint)
        job.init()
        job.doc.setdefault("done", False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
