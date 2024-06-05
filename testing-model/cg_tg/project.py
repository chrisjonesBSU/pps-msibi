"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help
"""
import signac
import pickle
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os
from unyt import Unit


class PPSCG(FlowProject):
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


@PPSCG.label
def initial_run_done(job):
    return job.doc.runs > 0


@PPSCG.label
def equilibrated(job):
    return job.doc.equilibrated


@PPSCG.label
def sampled(job):
    return job.doc.sampled


@PPSCG.label
def production_done(job):
    return job.isfile("production-restart.gsd")


def get_ref_values(job):
    ref_length = 0.3438 * Unit("nm")
    ref_mass = 32.06 * Unit("amu")
    ref_energy = 1.7782 * Unit("kJ/mol")
    ref_values_dict = {
        "length": ref_length,
        "mass": ref_mass,
        "energy": ref_energy
    }
    return ref_values_dict


def make_cg_system_bulk(job):
    from flowermd.base import Pack 
    from flowermd.library import PPS 

    chains = PPS(num_mols=job.sp.num_mols, lengths=job.sp.lengths)
    chains.coarse_grain(beads={"A": "c1cc(S)ccc1"})
    ref_values = get_ref_values(job)
    system = Pack(
            molecules=chains,
            density=job.sp.density,
            base_units=ref_values
    )
    return system


def make_cg_system_single_chain(job):
    from flowermd.base import System
    from flowermd.library import PPS 
    import mbuild as mb
    import numpy as np

    class SingleChainSystem(System):
        def __init__(self, molecules, base_units=dict()):
            super(SingleChainSystem, self).__init__(
                    molecules=molecules,
                    base_units=base_units
            )

        def _build_system(self):
            chain = self.all_molecules[0]
            head = chain.children[0]
            tail = chain.children[-1]
            chain_length = np.linalg.norm(tail.pos - head.pos)
            box = mb.Box(lengths=np.array([chain_length] * 3) * 1.05)
            comp = mb.Compound()
            comp.add(chain)
            comp.box = box
            chain.translate_to((box.Lx / 2, box.Ly / 2, box.Lz / 2))
            return comp

    chains = PPS(num_mols=job.sp.num_mols, lengths=job.sp.lengths)
    chains.coarse_grain(beads={"A": "c1cc(S)ccc1"})
    ref_values = get_ref_values(job)
    system = SingleChainSystem(molecules=chains, base_units=ref_values)
    return system


def get_ff(job):
    """"""
    msibi_project = signac.get_project(job.sp.msibi_project)
    msibi_job = msibi_project.open_job(id=job.sp.msibi_job)
    with open(msibi_job.fn("pps-msibi.pickle"), "rb") as f:
        hoomd_ff = pickle.load(f)
    return hoomd_ff


@PPSCG.post(initial_run_done)
@PPSCG.operation(
    directives={"ngpu": 1, "executable": "python -u"}, name="run"
)
def run(job):
    """Run initial single-chain simulation."""
    import unyt
    from unyt import Unit
    import flowermd
    from flowermd.base import Simulation
    from flowermd.utils import get_target_box_mass_density
    import hoomd
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        
        system = make_cg_system_bulk(job) 
        hoomd_ff = get_ff(job)
        for force in hoomd_ff:
            if isinstance(force, hoomd.md.bond.Table):
                if job.sp.harmonic_bonds:
                    print("Replacing bond table potential with harmonic")
                    hoomd_ff.remove(force)
                    harmonic_bond = hoomd.md.bond.Harmonic()
                    harmonic_bond.params["A-A"] = dict(k=1777.6, r0=1.4226)
                    hoomd_ff.append(harmonic_bond)
            else:
                pass
        # Store reference units and values
        job.doc.ref_mass = system.reference_mass.to("amu").value
        job.doc.ref_mass_units = "amu"
        job.doc.ref_energy = system.reference_energy.to("kJ/mol").value
        job.doc.ref_energy_units = "kJ/mol"
        job.doc.ref_length = (
                system.reference_length.to("nm").value
        )
        job.doc.ref_length_units = "nm"
        # Set up Simulation obj
        gsd_path = job.fn(f"trajectory{job.doc.runs}.gsd")
        log_path = job.fn(f"log{job.doc.runs}.txt")

        sim = Simulation(
            initial_state=system.hoomd_snapshot,
            forcefield=hoomd_ff,
            reference_values=system.reference_values,
            dt=job.sp.dt,
            gsd_write_freq=job.sp.gsd_write_freq,
            gsd_file_name=gsd_path,
            log_write_freq=job.sp.log_write_freq,
            log_file_name=log_path,
            seed=job.sp.sim_seed,
        )
        sim.pickle_forcefield(job.fn("forcefield.pickle"))
        # Store more unit information in job doc
        tau_kT = job.sp.dt * job.sp.tau_kT
        job.doc.tau_kT = tau_kT
        job.doc.real_time_step = sim.real_timestep.to("fs").value
        job.doc.real_time_units = "fs"
        target_box = get_target_box_mass_density(
                mass=system.mass.to("g"),
                density=job.sp.density * Unit("g/cm**3")
        )
        job.doc.target_box = target_box.value
        shrink_kT_ramp = sim.temperature_ramp(
                n_steps=job.sp.shrink_n_steps,
                kT_start=job.sp.shrink_kT,
                kT_final=job.sp.kT
        )
        sim.run_update_volume(
                final_box_lengths=target_box,
                n_steps=job.sp.shrink_n_steps,
                period=job.sp.shrink_period,
                tau_kt=tau_kT,
                kT=shrink_kT_ramp
        )
        sim.run_NVT(n_steps=job.sp.n_steps, kT=job.sp.kT, tau_kt=tau_kT)
        sim.save_restart_gsd(job.fn("restart.gsd"))
        job.doc.runs = 1
        print("Simulation finished.")


@PPSCG.pre(initial_run_done)
@PPSCG.post(equilibrated)
@PPSCG.operation(
    directives={"ngpu": 1, "executable": "python -u"},
    name="run-longer"
)
def run_longer(job):
    import unyt
    from unyt import Unit
    import flowermd
    from flowermd.base import Simulation
    import hoomd
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting and continuing simulation...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            hoomd_ff = pickle.load(f)

        gsd_path = job.fn(f"trajectory{job.doc.runs}.gsd")
        log_path = job.fn(f"log{job.doc.runs}.txt")
        ref_values = get_ref_values(job)
        sim = Simulation(
            initial_state=job.fn("restart.gsd"),
            forcefield=hoomd_ff,
            reference_values=ref_values,
            dt=job.sp.dt,
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


@PPSCG.pre(equilibrated)
@PPSCG.post(sampled)
@PPSCG.operation(
    directives={"ngpu": 1, "executable": "python -u"},
    name="production"
)
def production_run(job):
    import unyt
    from unyt import Unit
    import flowermd
    from flowermd.base import Simulation
    import hoomd
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        print("Restarting and continuing simulation...")
        print("Running the production run...")
        with open(job.fn("forcefield.pickle"), "rb") as f:
            hoomd_ff = pickle.load(f)

        gsd_path = job.fn(f"production.gsd")
        log_path = job.fn(f"production.txt")
        ref_values = get_ref_values(job)
        sim = Simulation(
            initial_state=job.fn("restart.gsd"),
            forcefield=hoomd_ff,
            reference_values=ref_values,
            dt=job.sp.dt,
            gsd_write_freq=int(5e5),
            gsd_file_name=gsd_path,
            log_write_freq=job.sp.log_write_freq,
            log_file_name=log_path,
            seed=job.sp.sim_seed,
        )
        print("Running simulation.")
        sim.run_NVT(
            n_steps=5e8,
            kT=job.sp.kT,
            tau_kt=job.doc.tau_kT,
        )
        sim.save_restart_gsd(job.fn("production-restart.gsd"))
        print("Simulation finished.")


@PPSCG.pre(production_done)
@PPSCG.post(sampled)
@PPSCG.operation(
    directives={"ngpu": 0, "executable": "python -u"},
    name="sample"
)
def sample(job):
    import numpy as np
    from cmeutils.dynamics import msd_from_gsd
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")
        steps_per_frame = int(5e5)
        # Update job doc
        job.doc.msd_n_samples = 15
        job.doc.msd_start_frame = 0
        job.doc.msd_chunk_size = 200 
        job.doc.msd_end_frame = 1000 - msd_chunk_size 
        job.doc.msd_start_indices = np.random.randint(
                job.doc.msd_start_frame,
                job.doc.msd_end_frame,
                job.doc.msd_n_samples
        ) 
        ts = job.doc.real_time_step * 1e-15
        ts_frame = steps_per_frame * ts
        for i in job.doc.msd_start_indices:
            msd = msd_from_gsd(
                    gsdfile=job.fn("production.gsd"),
                    start=int(i),
                    stop=int(i) + job.doc.msd_chunk_size,
                    atom_types="all",
                    msd_mode="direct"
            )
            msd_results = np.copy(msd.msd)
            conv_factor = (job.doc.ref_length**2) * 1e-18 
            job.doc.msd_units = "nm**2 / s"
            msd_results *= conv_factor 
            time_array = np.arrange(0, job.doc.msd_chunk_size, 1) * ts_frame
            np.save(file=job.fn(f"msd_time{i}.npy"), arr=time_array) 
            np.save(file=job.fn(f"msd_data_real{i}.npy"), arr=msd_results)
            np.save(file=job.fn(f"msd_data_raw{i}.npy"), arr=msd.msd)
            print(f"MSD calculation number {i} finished and saved...")

        print("Finished.")
        job.doc.sampled = True


if __name__ == "__main__":
    PPSCG(environment=Fry).main()
