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


def get_parameters(ordered_dict=OrderedDict()):
    '''Use the listed parameters below to set up
    your MSIBI instructions.

    '''
    parameters = ordered_dict

    # Optimizer parameters
    parameters["thermostat_tau"] = [0.03]
    parameters["dt"] = [0.0003]
    parameters["r_cut"] = [
            6.0,
    ]
    parameters["nlist_exclusions"] = [
            ["bond", "angle"],
            ["bond"],
    ]
    parameters["n_steps"] = [
            [1e6],
    ]
    parameters["state_alphas"] = [
        [[0.8]]
                #[[1.0, 0.0, 0.0, 0.0]],
                #[[0.0, 1.0, 0.0, 0.0]],
                #[[0.0, 0.0, 1.0, 0.0]],
                #[[0.0, 0.0, 0.0, 1.0]],
    ]
    parameters["n_iterations"] = [
            [20],
    ]
    parameters["epsilon"] = [1.25]
    parameters["T_scale"] = [1]
    parameters["sigma"] = [1.5]

    # State parameters
    # For each state: path to project with target trajectories,
    # and the job ID state point from pair_target_project.
    parameters["states"] = [
        [
            #{"name": "Ordered",
            # "alpha": 0.6,
            # "n_frames": 50,
            # "target_project": "/home/erjank_project/PPS-MSIBI/pps-msibi/validation",
            # "target_job_id": "100888a4bbe8114d13b7c682ba77a678",
            # "cg_file_name": "target_2monomer_per_bead.gsd",
            # },

            #{"name": "AmorphousTg",
            # "alpha": 0.6,
            # "n_frames": 50,
            # "target_project": "/home/erjank_project/PPS-MSIBI/pps-msibi/validation",
            # "target_job_id": "96282e58fc96112640e1330198828903",
            # "cg_file_name": "target_1monomer_per_bead.gsd",
            # },

            {"name": "Melted",
             "alpha": 0.6,
             "n_frames": 50,
             "target_project": "/home/erjank_project/PPS-MSIBI/pps-msibi/validation",
             "target_job_id": "212e36add95bb5744b05aa8ce8d29449",
             "cg_file_name": "target_1monomer_per_bead.gsd",
             },

            #{"name": "LowDen",
            # "alpha": 0.6,
            # "n_frames": 50,
            # "target_project": "/home/erjank_project/PPS-MSIBI/pps-msibi/training-runs/low-density-state",
            # "target_job_id": "5edb80880ecdb5cbd693d877e40882d5",
            # "cg_file_name": "target_1monomer_per_bead.gsd",
            # },
        ],
    ]

    # Pair parameters
    parameters["head_correction"] = ["linear"]
    parameters["pairs_nbins"] = [100]
    parameters["pairs"] = [
            {"type1": "A",
             "type2": "A",
             "epsilon": 1,
             "sigma": 1.5,
             "r_min": 0.1,
             "smoothing_window": 5,
             },
    ]
    parameters["smoothing_window"] = [5]
    # Bond parameters
    parameters["bond_project_path"] = [
            "/home/erjank_project/PPS-MSIBI/pps-msibi/msibi-flow/2monomers-per-bead/bond-flow"
    ]
    parameters["bond_job_id"] = ["c58e59d88849d8bfc1b89c62a2eeed64"]
    parameters["bonds"] = [
            {"type1": "A",
             "type2": "A",
             "file_path": "A-A_bond.csv",
             },
    ]
    # Angle parameters
    parameters["angle_project_path"] = [
            "/home/erjank_project/PPS-MSIBI/pps-msibi/msibi-flow/2monomers-per-bead/angle-flow"
    ]
    parameters["angle_job_id"] = ["88ec06d17a74cc92b229d529cb4646d7"]
    parameters["angles"] = [
            {"type1": "A",
             "type2": "A",
             "type3": "A",
             "file_path": "A-A-A_angle.csv",
             },
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
