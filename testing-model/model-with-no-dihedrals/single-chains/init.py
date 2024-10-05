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
import flow
import logging
from collections import OrderedDict
from itertools import product


def get_parameters():
    ''''''
    parameters = OrderedDict()
    parameters["num_mols"] = [1]
    parameters["lengths"] = [
            20,
            40,
            60,
            80,
            100,
            120,
            150,
            200,
            250,
            300,
            350,
            400,
            450,
            500,
            550,
            600,
            650,
            700
    ]
    parameters["density"] = [0.0001]
    parameters["periodic_dihedrals"] = [True]
    parameters["harmonic_bonds"] = [True]
    parameters["sigma_scale"] = [0.955]
    parameters["kT"] = [
            5.0,
            6.0,
            7.0,
            8.0,
    ]
    parameters["n_steps"] = [5e7]
    parameters["r_cut"] = [4.0]
    parameters["dt"] = [0.0003]
    parameters["tau_kT"] = [100]
    parameters["gsd_write_freq"] = [1e5]
    parameters["log_write_freq"] = [1e4]
    parameters["sim_seed"] = [42]
    # Get FF from the MSIBI Project
    parameters["msibi_project"] = [
        "/home/erjank_project/PPS-MSIBI/pps-msibi/msibi-flow/model-with-no-dihedrals/angle-flow-with-pairs"
    ]
    parameters["msibi_job"] = ["34c9e9f8fa7d942743adbf6835395671"]
    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project()
    param_names, param_combinations = get_parameters()
    # Create workspace of jobs
    for params in param_combinations:
        statepoint = dict(zip(param_names, params))
        job = project.open_job(statepoint)
        job.init()
        job.doc.setdefault("equilibrated", False)
        job.doc.setdefault("sampled", False)
        job.doc.setdefault("runs", 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
