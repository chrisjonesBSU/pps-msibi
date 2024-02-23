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
            2,
            20,
            40,
            60,
            80
    ]
    parameters["density"] = [0.0001]
    parameters["remove_hydrogens"] = [
            True,
            False
    ]
    parameters["remove_charges"] = [
            #True,
            False
    ]
    parameters["sigma_scale"] = [0.955]
    parameters["kT"] = [
            5.0,
            6.0,
            7.0,
            8.0,
    ]
    parameters["n_steps"] = [5e7]
    parameters["r_cut"] = [2.5]
    parameters["tau_kT"] = [100]
    parameters["gsd_write_freq"] = [1e5]
    parameters["log_write_freq"] = [1e4]
    parameters["sim_seed"] = [42]
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
        job.doc.setdefault("runs", 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
