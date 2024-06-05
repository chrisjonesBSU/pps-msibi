import matplotlib.pyplot as plt
import numpy as np
import signac
from cmeutils.sampling import equil_sample


def check_job_for_log_equilibrium(
		job,
		trim_cut,
		threshold_fraction=0.25,
		threshold_neff=50,
		value="mdcomputeThermodynamicQuantitiespotential_energy"
):
    log_path = job.fn(f"log{job.doc.runs - 1}.txt")
    all_data = np.genfromtxt(log_path, names=True)
    sample_data = all_data[value]
    try:
        uncorr_sample, uncorr_indices, prod_start, Neff = equil_sample(
            sample_data[trim_cut:],
            threshold_fraction=threshold_fraction,
            threshold_neff=threshold_neff
        )
        # Job is equilibrated
        job.doc.equilibrated = True
        # Starting index to use for log file
        job.doc.log_equil_start = int(trim_cut + prod_start)
        # Starting time step to use when sampling
        job.doc.equil_step_start = int(job.doc.log_equil_start * job.sp.log_write_freq)
        # Number of equilibrated samples
        job.doc.log_equil_Neff = int(Neff)
        # Index stride to use when sampling from log file
        job.doc.equil_log_stride = int(uncorr_indices[1] - uncorr_indices[0])
        # Time step stride to use when sampling
        job.doc.equil_step_stride = int(job.doc.equil_log_stride * job.sp.log_write_freq)
        if job.doc.equil_step_stride > job.sp.gsd_write_freq:
            job.doc.equil_gsd_stride = int(job.doc.equil_step_stride // job.sp.gsd_write_freq)
        else:
            job.doc.equil_gsd_stride = 1
        if job.doc.equil_step_start > job.sp.gsd_write_freq:
            job.doc.equil_gsd_start = int(job.doc.equil_step_start // job.sp.gsd_write_freq)
        else:
            job.doc.equil_gsd_start = 1
    except ValueError:
        print("Not equilibrated:")
        print(job.id)
        print()
