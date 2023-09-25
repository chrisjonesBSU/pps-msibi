"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os


class MyProject(FlowProject):
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


class R2(DefaultSlurmEnvironment):
    hostname_pattern = "r2"
    template = "r2.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition",
            default="shortgpuq",
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

# Definition of project-related labels (classification)
@MyProject.label
def sim_done(job):
    return job.doc.sim_done


@MyProject.label
def sample_done(job):
    return job.doc.sample_done


@MyProject.post(sim_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="npt"
)
def run_npt(job):
    import unyt
    from unyt import Unit
    import hoomd_organics
    from hoomd_organics.base.system import Pack
    from hoomd_organics.library import PPS, OPLS_AA_PPS
    from hoomd_organics.base.simulation import Simulation
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")

        density = job.sp.state_points.get("density")
        temp = job.sp.state_points.get("kT")
        job.doc.density = density
        job.doc.reduced_temp = temp

        pps = PPS(num_mols=job.sp.num_mols, lengths=job.sp.lengths)

        system = Pack(molecules=pps, density=job.sp.density)
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
        gsd_path = job.fn("trajectory.gsd")
        log_path = job.fn("log.txt")
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
        target_box = system.target_box / job.doc.ref_length 
        # Store more unit information in job doc
        tau_kT = job.doc.dt * job.sp.tau_kT
        tau_pressure = job.doc.dt * job.sp.tau_pressure
        job.doc.tau_kT = tau_kjob.doc.T
        job.doc.tau_pressure = tau_pressure
        job.doc.real_time_step = sim.real_timestep.to("fs").value
        job.doc.real_time_units = "fs"

        # Set up stuff for shrinking volume step
        print("Running shrink step.")
        shrink_kT_ramp = sim.temperature_ramp(
                n_steps=job.sp.shrink_n_steps,
                kT_start=job.sp.shrink_kT,
                kT_final=temp
        )

        # Anneal to just below target density
        sim.run_update_volume(
                final_box_lengths=target_box*1.10,
                n_steps=job.sp.shrink_n_steps,
                period=job.sp.shrink_period,
                tau_kt=tau_kT,
                kT=temp
        )

        # Run for a bit at lower density
        sim.run_NVT(n_steps=5e7, kT=temp, tau_kt=tau_kT)

        # Compress
        sim.run_update_volume(
                final_box_lengths=target_box*0.90,
                n_steps=1e7,
                period=1000,
                tau_kt=tau_kT,
                kT=job.sp.kT
        )

        # Expand back to target density
        sim.run_update_volume(
                final_box_lengths=target_box,
                n_steps=1e7,
                period=1000,
                tau_kt=tau_kT,
                kT=job.sp.kT
        )
        # Short run at NVT
        print("Shrinking and compressing finished.")
        print("Running NPT simulation.")
        sim.run_NVT(n_steps=job.doc.n_steps, kT=temp, tau_kt=tau_kT)
        sim.save_restart_gsd(job.fn("restart.gsd"))
        print("Simulation finished.")

@MyProject.pre(sim_done)
@MyProject.post(sample_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="sample-npt"
)
def sample_npt(job):
    # Add package imports here
    with job:
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        # Add your script here


if __name__ == "__main__":
    MyProject().main()
