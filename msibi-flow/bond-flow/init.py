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

import signac
import logging
from collections import OrderedDict
from itertools import product


def get_parameters():
    '''Use the listed parameters below to set up
    your MSIBI instructions.

    '''
    parameters = OrderedDict()

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
    parameters["single_chain_path"] = ["/home/erjank_project/PPS-MSIBI/pps-msibi/training-runs/single-chains"]
    parameters["states"] = [
        # Evenly Weighted, with 1.0
        [
            {"name": "A",
             "kT": 6.37,
             "target_trajectory": "1.27den-6.37kT-ua.gsd",
             "max_frames": 20,
             "alpha": 1.0,
             "exclude_bonded": True
             },

            {"name": "B",
             "kT": 4.2,
             "target_trajectory": "1.27den-4.2kT-ua.gsd",
             "max_frames": 20,
             "alpha": 1.0,
             "exclude_bonded": True
             },

            {"name": "C",
             "kT": 6.5,
             "target_trajectory": "single-chain.gsd",
             "max_frames": 200,
             "alpha": 1.0,
             "exclude_bonded": False
             },

            {"name": "D",
             "kT": 2.77,
             "target_trajectory": "1.40den-2.77kT-ua.gsd",
             "max_frames": 20,
             "alpha": 1.0,
             "exclude_bonded": True
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
    project = signac.init_project() # Set the signac project name
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
