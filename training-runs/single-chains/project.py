"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os
from unyt import Unit


class PPSSingleChain(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortgpu",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="batch",
            help="Specify the partition to submit to."
        )


@PPSSingleChain.label
def initial_run_done(job):
    return job.doc.runs > 0


@PPSSingleChain.label
def equilibrated(job):
    return job.doc.equilibrated


@PPSSingleChain.label
def sampled(job):
    return job.doc.sampled


def get_ref_values(job):
    ref_length = job.doc.ref_length * Unit(job.doc.ref_length_units)
    ref_mass = job.doc.ref_mass * Unit(job.doc.ref_mass_units)
    ref_energy = job.doc.ref_energy * Unit(job.doc.ref_energy_units)
    ref_values_dict = {
        "length": ref_length,
        "mass": ref_mass,
        "energy": ref_energy
    }
    return ref_values_dict


@PPSSingleChain.post(initial_run_done)
@PPSSingleChain.operation(
    directives={"ngpu": 1, "executable": "python -u"}, name="run"
)
def run(job):
    """Run initial single-chain simulation."""
    import unyt
    from unyt import Unit
    import flowermd
    from flowermd.base.system import Pack
    from flowermd.library import PPS, OPLS_AA_PPS
    from flowermd.base import Simulation
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")

        pps = PPS(num_mols=job.sp.num_mols, lengths=job.sp.lengths)
        system = Pack(
            molecules=pps,
            density=job.sp.density,
            packing_expand_factor=1
        )
        system.apply_forcefield(
            r_cut=job.sp.r_cut,
            auto_scale=True,
            scale_charges=True,
            remove_hydrogens=job.sp.remove_hydrogens,
            remove_charges=job.sp.remove_charges,
            force_field=OPLS_AA_PPS()
        )
        # Store reference units and values
        job.doc.ref_mass = system.reference_mass.to("amu").value
        job.doc.ref_mass_units = "amu"
        job.doc.ref_energy = system.reference_energy.to("kJ/mol").value
        job.doc.ref_energy_units = "kJ/mol"
        job.doc.ref_length = (
                system.reference_length.to("nm").value * job.sp.sigma_scale
        )
        job.doc.ref_length_units = "nm"
        if job.sp.remove_hydrogens:
            dt = 0.0003
        else:
            dt = 0.0001
        job.doc.dt = dt
        # Set up Simulation obj
        gsd_path = job.fn(f"trajectory{job.doc.runs}.gsd")
        log_path = job.fn(f"log{job.doc.runs}.txt")

        sim = Simulation.from_system(
            system,
            gsd_write_freq=job.sp.gsd_write_freq,
            gsd_file_name=gsd_path,
            log_write_freq=job.sp.log_write_freq,
            log_file_name=log_path,
            dt=job.doc.dt,
            seed=job.sp.sim_seed,
        )
        sim.pickle_forcefield(job.fn("forcefield.pickle"))
        sim.reference_length *= job.sp.sigma_scale

        # Store more unit information in job doc
        tau_kT = job.doc.dt * job.sp.tau_kT
        job.doc.tau_kT = tau_kT
        job.doc.real_time_step = sim.real_timestep.to("fs").value
        job.doc.real_time_units = "fs"

        sim.run_NVT(n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=tau_kT)
        sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.runs = 1
        print("Simulation finished.")


@PPSSingleChain.pre(initial_run_done)
@PPSSingleChain.post(equilibrated)
@PPSSingleChain.operation(
    directives={"ngpu": 1, "executable": "python -u"},
    name="run-longer"
)
def run_longer(job):
    import pickle
    import unyt
    from unyt import Unit
    import flowermd
    from flowermd.base import Simulation
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting and continuing simulation...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            ff = pickle.load(f)

        gsd_path = job.fn(f"trajectory-npt{job.doc.npt_runs}.gsd")
        log_path = job.fn(f"log-npt{job.doc.npt_runs}.txt")
        ref_values = get_ref_values(job)
        sim = Simulation(
            initial_state=job.fn("restart.gsd"),
            forcefield=ff,
            reference_values=ref_values,
            dt=job.doc.dt,
            gsd_write_freq=job.sp.gsd_write_freq,
            gsd_file_name=gsd_path,
            log_write_freq=job.sp.log_write_freq,
            log_file_name=log_path,
            seed=job.sp.sim_seed,
        )
        print("Running simulation.")
        sim.run_NVT(
            n_steps=5e7,
            kT=job.sp.kT,
            tau_kt=job.doc.tau_kT,
        )
        sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.runs += 1
        print("Simulation finished.")


@PPSSingleChain.pre(equilibrated)
@PPSSingleChain.post(sampled)
@PPSSingleChain.operation(
    directives={"ngpu": 0, "executable": "python -u"},
    name="sample"
)
def sample(job):
    import numpy as np
    import unyt
    from unyt import Unit
    from cmeutils.polymers import (
            radius_of_gyration,
            end_to_end_distance,
            persistence_length
    )
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Sampling Radius of Gyration...")
        print("------------------------------------")
        gsd_path = job.fn("target_1monomer_per_bead.gsd")
        rg_means, rg_std, rg_array = radius_of_gyration(
                gsd_file=gsd_path,
                start=job.doc.equil_gsd_start,
                stop=-1,
                stride=job.doc.equil_gsd_stride,
        )
        job.doc.rg_avg = np.mean(rg_means)
        job.doc.rg_std = np.std(rg_means)
        np.save(arr=np.array(rg_array), file=job.fn("rg_samples.npy"))
        print("------------------------------------")
        print("Sampling End-to-End Distance...")
        print("------------------------------------")
        re_means, re_std, re_array, re_vectors = end_to_end_distance(
                gsd_file=job.fn(f"trajectory{job.doc.runs - 1}.gsd"),
                start=job.doc.equil_gsd_start,
                stop=-1,
                stride=job.doc.equil_gsd_stride,
                head_index=0,
                tail_index=-1,
        )
        job.doc.re_avg = np.mean(re_means)
        job.doc.re_std = np.std(re_means)
        np.save(arr=np.array(re_array), file=job.fn("re_samples.npy"))
        print("------------------------------------")
        print("Sampling Persistence Length...")
        print("------------------------------------")
        lp_mean, lp_std = persistence_length(
                gsd_file=job.fn(f"trajectory{job.doc.runs - 1}.gsd"),
                select_atoms_arg = "name A A",
                start=job.doc.equil_gsd_start,
                window_size=25,
                stop=-1
        )
        job.doc.lp_mean = lp_mean
        job.doc.lp_std = lp_std

        print("Finished.")
        job.doc.sampled = True


if __name__ == "__main__":
    PPSSingleChain(environment=Fry).main()
