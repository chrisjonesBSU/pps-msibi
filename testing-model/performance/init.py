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
    parameters["x_len"] = [1.2]
    parameters["y_len"] = [1.2]
    parameters["n_repeats"] = [
        5,
        10,
        15,
        25,
        35,
        50,
        60,
        70,
        80,
        90,
        100,
    ]
    parameters["lengths"] = [40]
    parameters["density"] = [1.30]
    parameters["harmonic_bonds"] = [True]
    parameters["kT"] = [1.0]
    parameters["n_steps"] = [1e5]
    parameters["r_cut"] = [4.0]
    parameters["dt"] = [0.0003]
    parameters["tau_kT"] = [100]
    parameters["gsd_write_freq"] = [1e5]
    parameters["log_write_freq"] = [1e2]
    parameters["sim_seed"] = [42]
    # Get FF from the MSIBI Project
    parameters["msibi_project"] = [
        "/home/erjank_project/PPS-MSIBI/pps-msibi/msibi-flow/angle-flow-with-pairs"
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
