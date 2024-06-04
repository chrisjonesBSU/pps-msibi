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


def make_cg_system_lattice(job):
    from flowermd.base import Lattice 
    from flowermd.library import PPS 

    num_mols = int((job.sp.n_repeats ** 2) * 2)
    job.doc.num_mols = num_mols
    chains = PPS(num_mols=job.doc.num_mols, lengths=job.sp.lengths)
    chains.coarse_grain(beads={"A": "c1cc(S)ccc1"})
    ref_values = get_ref_values(job)
    system = Lattice(
            molecules=chains,
            n=job.sp.n_repeats,
            x=job.sp.x_len,
            y=job.sp.y_len,
            base_units=ref_values
    )
    return system


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
        
        system = make_cg_system_lattice(job) 
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

        gsd_path = job.fn(f"trajectory-npt{job.doc.npt_runs}.gsd")
        log_path = job.fn(f"log-npt{job.doc.npt_runs}.txt")
        ref_values = get_ref_values(job)
        sim = Simulation(
            initial_state=job.fn("npt-restart.gsd"),
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
        sim.run_NPT(
                n_steps=job.sp.n_steps,
                kT=job.sp.kT,
                pressure=job.sp.pressure,
                tau_pressure=job.doc.tau_pressure,
                tau_kt=job.doc.tau_kT,
                gamma=job.sp.gamma,
        )
        sim.save_restart_gsd(job.fn("npt-restart.gsd"))
        job.doc.runs += 1
        print("Simulation finished.")


@PPSCG.pre(equilibrated)
@PPSCG.post(sampled)
@PPSCG.operation(
    directives={"ngpu": 0, "executable": "python -u"},
    name="sample"
)
def sample(job):
    import numpy as np
    import unyt
    from unyt import Unit
    with job:
        print("------------------------------------")
        print("JOB ID NUMBER:")
        print(job.id)
        print("------------------------------------")

        print("Finished.")
        job.doc.sampled = True


if __name__ == "__main__":
    PPSCG(environment=Fry).main()
