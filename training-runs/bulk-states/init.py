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
    parameters["num_mols"] = [50]
    parameters["lengths"] = [30]
    parameters["density"] = [1.32]
    parameters["remove_hydrogens"] = [
            #True,
            False
    ]
    parameters["remove_charges"] = [
            #True,
            False
    ]
    parameters["sigma_scale"] = [0.96]
    parameters["kT"] = [
            1.5, # Just below Tg
            2.2, # About half-way between Tg and Tm
            2.5, # Just below Tm
            2.8, # Above Tm (Processing temp of 320 C)
    ]
    parameters["pressure"] = [0.002332]
    parameters["n_steps"] = [5e7]
    parameters["shrink_kT"] = [8.0]
    parameters["shrink_n_steps"] = [5e7]
    parameters["shrink_period"] = [10000]
    parameters["r_cut"] = [2.5]
    parameters["tau_kT"] = [100]
    parameters["tau_pressure"] = [10]
    parameters["gsd_write_freq"] = [1e5]
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
        job.doc.setdefault("sim_done", False)
        job.doc.setdefault("sample_done", False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
