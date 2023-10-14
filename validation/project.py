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
def sample_volume_done(job):
    return job.doc.volume_sampled


def sample_msd_done(job):
    return job.doc.msd_sampled


@MyProject.label
def initial_npt_run_done(job):
    return job.doc.npt_runs >= 1


@MyProject.label
def initial_nvt_run_done(job):
    return job.doc.nvt_runs >= 1


@MyProject.label
def npt_equilibrated(job):
    return job.doc.npt_equilibrated


@MyProject.label
def nvt_equilibrated(job):
    return job.doc.nvt_equilibrated

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


@MyProject.post(initial_npt_run_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="npt"
)
def run_npt(job):
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
        if job.sp.sigma_scale == 1.0:
            job.doc.pressure = 0.0015996
        elif job.sp.sigma_scale == 0.955:
            job.doc.pressure = 0.0013933
        if job.sp.remove_hydrogens:
            dt = 0.0003
        else:
            dt = 0.0001
        job.doc.dt = dt
        # Set up Simulation obj
        gsd_path = job.fn(f"trajectory-npt{job.doc.npt_runs}.gsd")
        log_path = job.fn(f"log-npt{job.doc.npt_runs}.txt")

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
        print("Shrinking and compressing finished.")
        # Short run at NVT
        print("Running NVT simulation.")
        sim.run_NVT(n_steps=1e7, kT=job.sp.kT, tau_kt=tau_kT)
        sim.save_restart_gsd(job.fn("restart.gsd"))
        print("Running NPT simulation.")
        sim.run_NPT(
            n_steps=job.sp.n_steps,
            kT=job.sp.kT,
            pressure=job.doc.pressure,
            tau_kt=tau_kT,
            tau_pressure=job.doc.tau_pressure,
            gamma=job.sp.gamma
        )
        sim.save_restart_gsd(job.fn("restart-npt.gsd"))
        job.doc.npt_runs = 1
        print("Simulation finished.")


@MyProject.pre(initial_npt_run_done)
@MyProject.post(npt_equilibrated)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"},
        name="run-npt-longer"
)
def run_npt_longer(job):
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
        print("Restarting and continuing NPT simulation...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            ff = pickle.load(f)


        gsd_path = job.fn(f"trajectory-npt{job.doc.npt_runs}.gsd")
        log_path = job.fn(f"log-npt{job.doc.npt_runs}.txt")
        ref_values = get_ref_values(job)
        sim = Simulation(
                initial_state=job.fn("restart-npt.gsd"),
                forcefield=ff,
                reference_values=ref_values,
                dt=job.doc.dt,
                gsd_write_freq=job.sp.gsd_write_freq,
                gsd_file_name=gsd_path,
                log_write_freq=job.sp.log_write_freq,
                log_file_name=log_path,
                seed=job.sp.sim_seed,
        )
        print("Running NPT simulation.")
        sim.run_NPT(
            n_steps=2e7,
            kT=job.sp.kT,
            pressure=job.doc.pressure,
            tau_kt=job.doc.tau_kT,
            tau_pressure=job.doc.tau_pressure,
            gamma=job.sp.gamma
        )
        sim.save_restart_gsd(job.fn("restart-npt.gsd"))
        job.doc.npt_runs += 1
        print("Simulation finished.")


@MyProject.pre(sample_volume_done)
@MyProject.post(initial_nvt_run_done)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"}, name="nvt"
)
def run_nvt(job):
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
        print("Running initial NVT simulation...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            ff = pickle.load(f)

        gsd_path = job.fn(f"trajectory-nvt{job.doc.nvt_runs}.gsd")
        log_path = job.fn(f"log-nvt{job.doc.nvt_runs}.txt")
        ref_values = get_ref_values(job)
        sim = Simulation(
                initial_state=job.fn("restart-npt.gsd"),
                forcefield=ff,
                reference_values=ref_values,
                dt=job.doc.dt,
                gsd_write_freq=job.sp.gsd_write_freq,
                gsd_file_name=gsd_path,
                log_write_freq=job.sp.log_write_freq,
                log_file_name=log_path,
                seed=job.sp.sim_seed,
        )
        sim.run_update_volume(
                n_steps=2e6,
                period=1,
                kT=job.sp.kT,
                tau_kt=job.doc.tau_kT,
                final_density=job.doc.avg_density
        )
        sim.run_NVT(n_steps=5e7, kT=job.sp.kT, tau_kt=job.doc.tau_kT)
        sim.save_restart_gsd(job.fn("restart-nvt.gsd"))
        job.doc.nvt_runs = 1
        print("Simulation finished.")


@MyProject.pre(initial_nvt_run_done)
@MyProject.post(nvt_equilibrated)
@MyProject.operation(
        directives={"ngpu": 1, "executable": "python -u"},
        name="run-nvt-longer"
)
def run_nvt_longer(job):
    import pickle
    import unyt
    from unyt import Unit
    import jankflow
    from jankflow.base.simulation import Simulation
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting NVT simulation...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            ff = pickle.load(f)

        gsd_path = job.fn(f"trajectory-nvt{job.doc.nvt_runs}.gsd")
        log_path = job.fn(f"log-nvt{job.doc.nvt_runs}.txt")
        ref_values = get_ref_values(job)
        sim = Simulation(
                initial_state=job.fn("restart-nvt.gsd"),
                forcefield=ff,
                reference_values=ref_values,
                dt=job.doc.dt,
                gsd_write_freq=job.sp.gsd_write_freq,
                gsd_file_name=gsd_path,
                log_write_freq=job.sp.log_write_freq,
                log_file_name=log_path,
                seed=job.sp.sim_seed,
        )
        sim.run_NVT(n_steps=1e7, kT=job.sp.kT, tau_kt=job.doc.tau_kT)
        sim.save_restart_gsd(job.fn("restart-nvt.gsd"))
        job.doc.nvt_runs += 1
        print("Simulation finished.")

if __name__ == "__main__":
    MyProject().main()
