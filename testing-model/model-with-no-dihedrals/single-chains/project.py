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
            default="v100," "batch",
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
    ref_length = 0.3438 * Unit("nm")
    ref_mass = 32.06 * Unit("amu")
    ref_energy = 1.065 * Unit("kJ/mol")
    ref_values_dict = {
        "length": ref_length,
        "mass": ref_mass,
        "energy": ref_energy
    }
    return ref_values_dict


def make_cg_system(job):
    from flowermd.base import System
    from flowermd.library import LJChain, PPS 
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
            box = mb.Box(lengths=np.array([chain_length] * 3) * 1.5)
            comp = mb.Compound()
            comp.add(chain)
            comp.box = box
            chain.translate_to((box.Lx / 2, box.Ly / 2, box.Lz / 2))
            return comp

    #chains = LJChain(
    #        num_mols=job.sp.num_mols,
    #        lengths=job.sp.lengths,
    #        bond_lengths={"A-A": 1.48}
    #)
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


@PPSSingleChain.post(initial_run_done)
@PPSSingleChain.operation(
    directives={"ngpu": 1, "executable": "python -u"}, name="run"
)
def run(job):
    """Run initial single-chain simulation."""
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
        
        system = make_cg_system(job) 
        hoomd_ff = get_ff(job)
        for force in hoomd_ff:
            if isinstance(force, hoomd.md.pair.Table):
                original_nlist = force.nlist
                original_exclusions = original_nlist.exclusions
                tree_nlist = hoomd.md.nlist.Tree(
                        buffer=0.4, exclusions=original_exclusions
                )
                force.nlist = tree_nlist 
            elif isinstance(force, hoomd.md.bond.Table):
                if job.sp.harmonic_bonds:
                    print("Replacing bond table potential with harmonic")
                    hoomd_ff.remove(force)
                    harmonic_bond = hoomd.md.bond.Harmonic()
                    harmonic_bond.params["A-A"] = dict(k=1777.6, r0=1.4226)
                    hoomd_ff.append(harmonic_bond)
        # Store reference units and values
        job.doc.ref_mass = system.reference_mass.to("amu").value
        job.doc.ref_mass_units = "amu"
        job.doc.ref_energy = system.reference_energy.to("kJ/mol").value
        job.doc.ref_energy_units = "kJ/mol"
        job.doc.ref_length = (
                system.reference_length.to("nm").value * job.sp.sigma_scale
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

        #sim.pickle_forcefield(job.fn("forcefield.pickle"))
        # Store more unit information in job doc
        tau_kT = job.sp.dt * job.sp.tau_kT
        job.doc.tau_kT = tau_kT
        job.doc.real_time_step = sim.real_timestep.to("fs").value
        job.doc.real_time_units = "fs"
        
        if job.sp.periodic_dihedrals:
                print("Adding a periodic dihedral potential")
                sim.run_NVT(n_steps=1e6, kT=job.sp.kT, tau_kt=tau_kT)
                periodic_dihedral = hoomd.md.dihedral.Periodic()
                periodic_dihedral.params["A-A-A-A"] = dict(k=8.0, phi0=0, d=-1, n=1)
                sim.add_force(periodic_dihedral)
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
        rg_means, rg_std, rg_array = radius_of_gyration(
                gsd_file=job.fn(f"trajectory{job.doc.runs - 1}.gsd"),
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
