"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
from flow import FlowProject
from flow.environment import DefaultSlurmEnvironment


class PPSProject(FlowProject):
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


@PPSProject.label
def system_initialized(job):
    return job.doc.system_initialized


@PPSProject.label
def initial_run_done(job):
    return job.doc.npt_runs >= 1


@PPSProject.label
def equilibrated(job):
    return job.doc.equilibrated


@PPSProject.post(system_initialized)
@PPSProject.operation(
    directives={"executable": "python -u"}, name="initiate"
)
def initiate_system(job):
    """Initialize the system and apply ff. Save snapshot and forcefield."""
    import numpy as np
    import mbuild as mb
    from flowermd.base.system import Lattice
    from flowermd.library import PPS, OPLS_AA_PPS
    from flowermd.base.simulation import Simulation
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")

        pps = PPS(num_mols=job.sp.num_mols, lengths=job.sp.lengths)
        n = int(np.sqrt(job.sp.num_mols // 2))
        system = Lattice(
            molecules=pps,
            y=0.867,
            x=0.561,
            n=n
        )
        print("initial box lengths: ", system.system.box.lengths)
        system.system.box = mb.box.Box(
            lengths=np.array(system.system.box.lengths) * (1, 1, 1.5),
            angles=(90, 90, 90))
        print("box lengths after changing z length: ",
              system.system.box.lengths)
        system.gmso_system = system._convert_to_gmso()
        print("applying ff...")
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
        tau_kT = job.doc.dt * job.sp.tau_kT
        tau_pressure = job.doc.dt * job.sp.tau_pressure
        job.doc.tau_kT = tau_kT
        job.doc.tau_pressure = tau_pressure
        job.doc.real_time_step = sim.real_timestep.to("fs").value
        job.doc.real_time_units = "fs"
        sim.save_restart_gsd(job.fn("initial_snap.gsd"))
        print("initial snapshot box: ", system.hoomd_snapshot.configuration.box)
        job.doc.system_initialized = True
        print("system initialized")


@PPSProject.post(initial_run_done)
@PPSProject.pre(system_initialized)
@PPSProject.operation(
    directives={"ngpu": 1, "executable": "python -u"}, name="lattice"
)
def run_validate_lattice(job):
    """Run a bulk simulation; equilibrate in NPT"""
    from unyt import Unit
    import numpy as np
    import pickle
    from flowermd.base.simulation import Simulation
    from flowermd.utils import get_target_box_mass_density
    from utils import check_npt_equilibration
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            ff = pickle.load(f)

        gsd_path = job.fn("trajectory.gsd")
        log_path = job.fn("log.txt")

        sim = Simulation(
            initial_state=job.fn("initial_snap.gsd"),
            forcefield=ff,
            dt=job.doc.dt,
            gsd_write_freq=job.sp.gsd_write_freq,
            gsd_file_name=gsd_path,
            log_write_freq=job.sp.log_write_freq,
            log_file_name=log_path,
            seed=job.sp.sim_seed,
        )
        # Step 1: Quick chain relaxation at cold temperature:
        sim.run_NVT(n_steps=1e5, kT=0.2, tau_kt=job.doc.tau_kT,
                    write_at_start=True)
        sim.flush_writers()

        # Step 2: Quick shrink to reach the target density
        n = int(np.sqrt(job.sp.num_mols // 2))
        mass = sim.mass.to('g')
        Lx = (n * 0.561 * Unit('nm')).to('cm')
        Ly = (n * 0.867 * Unit('nm')).to('cm')
        density = job.sp.density * Unit("g") / (Unit("cm") ** 3)
        reference_length = job.doc.ref_length

        target_box = get_target_box_mass_density(density=density, mass=mass,
                                                 x_constraint=Lx,
                                                 y_constraint=Ly)
        target_box = target_box.to("nm") / reference_length
        print('target box: ', target_box)
        sim.run_update_volume(final_box_lengths=target_box.value, n_steps=500,
                              period=1, kT=0.2, tau_kt=job.doc.tau_kT,
                              write_at_start=True)
        sim.flush_writers()

        # Step 3: Quick chain relaxation at cold temperature
        sim.run_NVT(n_steps=1e5, kT=0.2, tau_kt=job.doc.tau_kT,
                    write_at_start=True)
        sim.flush_writers()

        # Step 4: NVT run where we ramp up from kT = 0.2 to kT = 1
        heating_ramp = sim.temperature_ramp(n_steps=1e5, kT_start=0.2,
                                            kT_final=job.sp.kT)
        sim.run_NVT(n_steps=5e5, kT=heating_ramp, tau_kt=job.doc.tau_kT,
                    write_at_start=True)
        sim.flush_writers()

        # step 5: Hold at kT=1.0 for a while
        sim.run_NVT(n_steps=1e6, kT=job.sp.kT, tau_kt=job.doc.tau_kT)
        sim.flush_writers()

        sim.save_restart_gsd(job.fn("restart.gsd"))

        print("Running NPT simulation.")
        sim.run_NPT(
            n_steps=job.sp.n_steps,
            kT=job.sp.kT,
            pressure=job.sp.pressure,
            tau_kt=job.doc.tau_kT,
            tau_pressure=job.doc.tau_pressure,
            gamma=job.sp.gamma
        )
        sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.npt_runs = 1
        npt_sample_count = int(job.sp.n_steps / job.sp.log_write_freq)
        job.doc.npt_sample_count = npt_sample_count
        is_equilibrated = check_npt_equilibration(job,
                                                  sample_idx=-npt_sample_count)
        if is_equilibrated:
            print("Simulation equilibrated.")
            job.doc.equilibrated = True
        print("Simulation finished.")


@PPSProject.post(equilibrated)
@PPSProject.pre(initial_run_done)
@PPSProject.operation(
    directives={"ngpu": 1, "executable": "python -u"}, name="run-longer"
)
def run_longer(job):
    import pickle
    from flowermd.base.simulation import Simulation
    from utils import check_npt_equilibration
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting and continuing a simulation...")
        run_num = job.doc.npt_runs + 1
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

        npt_sample_count = int(
            job.sp.n_steps / job.sp.log_write_freq) + job.doc.npt_sample_count
        is_equilibrated = check_npt_equilibration(job,
                                                  sample_idx=-npt_sample_count)
        job.doc.npt_sample_count = npt_sample_count
        if is_equilibrated:
            print("Simulation equilibrated.")
            job.doc.equilibrated = True
        print("Simulation finished.")


if __name__ == "__main__":
    PPSProject(environment=Fry).main()
