import numpy as np
from cmeutils.sampling import is_equilibrated


def combine_log_files(job, ensemble="npt",
                      value="mdcomputeThermodynamicQuantitiesvolume"):
    arrays = []
    if ensemble == "npt":
        n_runs = job.doc.npt_runs
    elif ensemble == "nvt":
        n_runs = job.doc.nvt_runs
    else:
        raise ValueError(f"Unknown ensemble {ensemble}")

    for i in range(n_runs):
        fpath = job.fn(f"log-{ensemble}{i}.txt")
        data = np.genfromtxt(fpath, names=True)
        data_array = data[value]
        arrays.append(data_array)
    return np.concatenate(arrays)


def check_npt_equilibration(job, sample_idx):
    volume = combine_log_files(job,
                               ensemble="npt",
                               value="mdcomputeThermodynamicQuantitiesvolume")
    potential_energy = combine_log_files(job,
                                         ensemble="npt",
                                         value="mdcomputeThermodynamicQuantitiespotential_energy")

    vol_eq = is_equilibrated(volume[sample_idx:],
                             threshold_fraction=0.15,
                             threshold_neff=200)[0]
    pe_eq = is_equilibrated(potential_energy[sample_idx:],
                            threshold_fraction=0.15,
                            threshold_neff=200)[0]
    return all([vol_eq, pe_eq])


def check_nvt_equilibration(job, sample_idx):
    potential_energy = combine_log_files(job,
                                         ensemble="nvt",
                                         value="mdcomputeThermodynamicQuantitiespotential_energy")

    pe_eq = is_equilibrated(potential_energy[sample_idx:],
                            threshold_fraction=0.15,
                            threshold_neff=200)[0]
    return pe_eq

