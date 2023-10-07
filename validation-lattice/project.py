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

@MyProject.label
def validate_tg_done(job):
    return job.doc.validate_tg_done


@MyProject.label
def sample_volume_done(job):
    return job.doc.volume_sampled


@MyProject.label
def initial_run_done(job):
    return job.doc.n_runs >= 1


@MyProject.label
def equilibrated(job):
    return job.doc.equilibrated


@MyProject.post(equilibrated)
@MyProject.pre(initial_run_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="run-longer"
)
def run_longer(job):
    import pickle

    import unyt
    from unyt import Unit
    import jankflow
    from jankflow.base.system import Pack
    from jankflow.library import PPS, OPLS_AA_PPS
    from jankflow.base.simulation import Simulation
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting and continuing a simulation...")
        run_num = job.doc.n_runs + 1
        with open(job.fn("forcefield.pickle"), "rb") as f:
            ff = pickle.load(f)

        gsd_path = job.fn(f"trajectory{run_num}.gsd")
        log_path = job.fn(f"log{run_num}.txt")

        sim = Simulation(
                initial_state=job.fn("restart.gsd"),
                forcefield=ff,
                dt=job.doc.dt,
                gsd_write_freq=job.sp.gsd_write_freq,
                gsd_file_name=gsd_path,
                log_write_freq=job.sp.log_write_freq,
                log_file_name=log_path,
                seed=job.sp.sim_seed,
        )
        print("Running NPT simulation.")
        sim.run_NPT(
            n_steps=1e7,
            kT=job.sp.kT,
            pressure=job.sp.pressure,
            tau_kt=job.doc.tau_kT,
            tau_pressure=job.doc.tau_pressure,
            gamma=job.sp.gamma
        )
        sim.save_restart_gsd(job.fn("restart.gsd"))
        print("Simulation finished.")


@MyProject.post(initial_run_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="validate-tg"
)
def run_validate_tg(job):
    """Run a bulk simulation; equilibrate in NPT"""
    import unyt
    from unyt import Unit
    import jankflow
    from jankflow.base.system import Pack
    from jankflow.library import PPS, OPLS_AA_PPS
    from jankflow.base.simulation import Simulation
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
        sim.reference_length *= job.sp.sigma_scale

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
                final_density=job.sp.density*1.10,
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
        sim.run_NVT(n_steps=1e7, kT=job.sp.kT, tau_kt=tau_kT)
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
        job.doc.n_runs = 1
        print("Simulation finished.")


@MyProject.post(initial_run_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="lattice"
)
def run_validate_lattice(job):
    """Run a bulk simulation; equilibrate in NPT"""
    import unyt
    from unyt import Unit
    import jankflow
    from jankflow.base.system import Lattice
    from jankflow.library import PPS, OPLS_AA_PPS
    from jankflow.base.simulation import Simulation
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")

        pps = PPS(num_mols=job.sp.num_mols, lengths=job.sp.lengths)

        system = Lattice(
                molecules=pps,
                density=job.sp.density,
                x=0.867,
                y=0.561,
                n=int(np.sqrt(job.sp.n_mols // 2))
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
        sim.reference_length *= job.sp.sigma_scale

        # Store more unit information in job doc
        target_box = system.target_box / job.doc.ref_length
        tau_kT = job.doc.dt * job.sp.tau_kT
        tau_pressure = job.doc.dt * job.sp.tau_pressure
        job.doc.tau_kT = tau_kT
        job.doc.tau_pressure = tau_pressure
        job.doc.real_time_step = sim.real_timestep.to("fs").value
        job.doc.real_time_units = "fs"

        #Quick chain relaxation at cold temperature:
        sim.run_NVT(n_steps=5e6, kT=0.8, tau_kt=tau_kT)

        print("Running shrink step.")
        heating_ramp = sim.temperature_ramp(
                n_steps=1e7,
                kT_start=0.8,
                kT_final=job.sp.kT
        )

        sim.run_NVT(n_steps=1e7, kT=heating_ramp, tau_kt=tau_kT)
        sim.run_NVT(n_steps=5e6, kT=job.sp.kT, tau_kt=tau_kT)
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
        job.doc.n_runs = 1
        print("Simulation finished.")

if __name__ == "__main__":
    MyProject().main()
