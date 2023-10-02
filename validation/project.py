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
def validate_tg_done(job):
    return job.doc.validate_tg_done


@MyProject.label
def sample_volume_done(job):
    return job.doc.volume_sampled


@MyProject.post(validate_tg_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="validate-tg"
)
def run_validate_tg(job):
    """Run a bulk simulation; equilibrate in NPT"""
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

        pps = PPS(num_mols=job.sp.num_mols, lengths=job.sp.lengths)

        system = Pack(
                molecules=pps, density=job.sp.density,
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

        # Store more unit information in job doc
        target_box = system.target_box / job.doc.ref_length
        tau_kT = job.doc.dt * job.sp.tau_kT
        tau_pressure = job.doc.dt * job.sp.tau_pressure
        job.doc.tau_kT = tau_kT
        job.doc.tau_pressure = tau_pressure
        job.doc.real_time_step = sim.real_timestep.to("fs").value
        job.doc.real_time_units = "fs"

        # Set up stuff for shrinking volume step
        print("Running shrink step.")
        shrink_kT_ramp = sim.temperature_ramp(
                n_steps=job.sp.shrink_n_steps,
                kT_start=job.sp.shrink_kT,
                kT_final=job.sp.kT
        )

        # Anneal to just below target density
        sim.run_update_volume(
                final_density=job.sp.density*1.15,
                n_steps=job.sp.shrink_n_steps,
                period=job.sp.shrink_period,
                tau_kt=tau_kT,
                kT=shrink_kT_ramp
        )

        # Expand back to target density
        sim.run_update_volume(
                final_density=job.sp.density,
                n_steps=1e7,
                period=500,
                tau_kt=tau_kT,
                kT=job.sp.kT
        )
        sim.save_restart_gsd(job.fn("restart.gsd"))
        print("Shrinking and compressing finished.")
        # Short run at NVT
        print("Running NVT simulation.")
        sim.run_NVT(n_steps=2e7, kT=job.sp.kT, tau_kt=tau_kT)
        sim.save_restart_gsd(job.fn("restart.gsd"))
        print("Running NPT simulation.")
        sim.run_NPT(
            n_steps=job.sp.n_steps,
            kT=job.sp.kT,
            pressure=job.sp.pressure,
            tau_kt=tau_kT,
            tau_pressure=job.doc.tau_pressure,
            gamma=job.sp.gamma
        )
        sim.save_restart_gsd(job.fn("restart.gsd"))
        print("Simulation finished.")

@MyProject.pre(validate_tg_done)
@MyProject.post(sample_volume_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="sample-npt"
)
def sample_npt(job):
    from cmeutils.sampling import is_equilibrated, equil_sample
    import numpy as np
    import unyt as u
    from unyt import Unit

    with job:
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        data = np.genfromtxt(job.fn("log.txt"), names=True)
        volume = data["mdcomputeThermodynamicQuantitiesvolume"]
        pe = data["mdcomputeThermodynamicQuantitiespotential_energy"]
        num_points = len(volume)
        volume_eq = is_equilibrated(
                volume[num_points//2:],
                threshold_neff=100,
                threshold_fraction=0.10
        )[0]
        pe_eq = is_equilibrated(
                pe[num_points//2:],
                threshold_neff=100,
                threshold_fraction=0.10
        )[0]
        if all([volume_eq, pe_eq]):
            job.doc.equilibrated = True
            uncorr_sample, uncorr_indices, prod_start, Neff = equil_sample(
                    volume[num_points//2:],
                    threshold_fraciton=0.10,
                    threshold_neff=100
            )
            vol_nm = uncorr_sample * job.doc.ref_length * Unit("nm**3")
            vol_cm = vol_nm.to("cm**3")
            np.savetxt(job.fn("vol_sample_indices.txt"), uncorr_indices)
            np.savetxt(job.fn("volume_cc.txt"), vol_cm.value)
            job.doc.avg_vol = np.mean(vol_cm).value
            job.doc.vol_std = np.std(vol_cm).value

            with gsd.hoomd.open(job.fn("restart.gsd")) as traj:
                snap = traj[0]
                reduced_mass = sum(snap.particles.mass)
                mass_amu = (reduced_mass * job.doc.ref_mass) * Unit("amu")
                mass_g = mass_amu.to("g")
                job.doc.mass_g = mass_g.value

            job.doc.avg_density = job.doc.mass_g / job.doc.avg_vol
            job.doc.density_std = job.doc.mass_g / job.doc.vol_std
            job.doc.volume_sampled = True

if __name__ == "__main__":
    MyProject().main()
