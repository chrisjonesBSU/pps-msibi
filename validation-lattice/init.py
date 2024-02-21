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
    parameters["num_mols"] = [500]
    parameters["lengths"] = [20]
    parameters["density"] = [1.43]
    parameters["remove_hydrogens"] = [
            True,
    ]
    parameters["remove_charges"] = [
            True,
            False
    ]
    parameters["sigma_scale"] = [1.0]
    parameters["kT"] = [1.0,]
    parameters["pressure"] = [0.0023263]
    parameters["n_steps"] = [5e6]
    parameters["r_cut"] = [2.5]
    parameters["tau_kT"] = [100]
    parameters["tau_pressure"] = [800]
    parameters["gamma"] = [0]
    parameters["gsd_write_freq"] = [1e4]
    parameters["log_write_freq"] = [1e4]
    parameters["sim_seed"] = [42]
    return list(parameters.keys()), list(product(*parameters.values()))


def main():
    project = signac.init_project() # Set the signac project name
    param_names, param_combinations = get_parameters()
    # Create the generate jobs
    for params in param_combinations:
        statepoint = dict(zip(param_names, params))
        job = project.open_job(statepoint)
        job.init()
        job.doc.setdefault("equilibrated", False)
        job.doc.setdefault("system_initialized", False)
        job.doc.setdefault("npt_runs", 0)
        job.doc.setdefault("npt_sample_count", 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
