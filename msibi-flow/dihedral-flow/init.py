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

import numpy as np
import signac


def get_parameters():
    '''Use the listed parameters below to set up
    your MSIBI instructions.

    '''
    parameters = OrderedDict()

    # Optimizer parameters
    parameters["nlist"] = ["Cell"]
    parameters["thermostat_tau"] = [0.03]
    parameters["dt"] = [0.0003]
    parameters["nlist_exclusions"] = [["bond", "angle"]]
    parameters["n_steps"] = [5e5]
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
             "n_frames": 100,
             },
        ],

    ]

    # Dihedral parameters
    parameters["head_correction"] = ["linear"]
    parameters["dihedrals_nbins"] = [60]
    parameters["dihedrals"] = [
        [
            {"type1": "A",
             "type2": "A",
             "type3": "A",
             "type4": "A",
             "x0": 0.0,
             "x_min": -np.pi,
             "x_max": np.pi,
             "k2": 100,
             "k3": 0,
             "k4": 0,
             },
        ]
    ]
    # Pair parameters
    parameters["pair_project_path"] = [
        "/home/erjank_project/PPS-MSIBI/pps-msibi/msibi-flow/pair-flow"]
    parameters["pair_job_id"] = [""]
    parameters["pairs_nbins"] = [100]
    parameters["pairs"] = [
        [
            {"type1": "A",
             "type2": "A",
             "r_cut": 4.0,
             "exclude_bonded": True,
             "file_path": "AA_pair.csv",
             },
        ]
    ]
    # Angle parameters
    parameters["angle_project_path"] = [
        "/home/erjank_project/PPS-MSIBI/pps-msibi/msibi-flow/angle-flow"]
    parameters["angle_job_id"] = [""]
    parameters["angles_nbins"] = [100]
    parameters["angles"] = [
        [
            {"type1": "A",
             "type2": "A",
             "type3": "A",
             "file_path": "AAA_angle.csv",
             },
        ]
    ]

    # Bond parameters
    parameters["bond_project_path"] = [
        "/home/erjank_project/PPS-MSIBI/pps-msibi/msibi-flow/bond-flow"]
    parameters["bond_job_id"] = ["ba28bd502cec3ae0056aef66e545b069"]
    parameters["bonds_nbins"] = [100]
    parameters["bonds"] = [
        [
            {"type1": "A",
             "type2": "A",
             "file_path": "AA_bond.csv",
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
