"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help

"""

import os

import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment


class BondMSIBI(FlowProject):
    pass


class Borah(DefaultSlurmEnvironment):
    hostname_pattern = "borah"
    template = "borah.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition", default="gpu",
            help="Specify the partition to submit to."
        )


class Fry(DefaultSlurmEnvironment):
    hostname_pattern = "fry"
    template = "fry.sh"

    @classmethod
    def add_args(cls, parser):
        parser.add_argument(
            "--partition", default="batch",
            help="Specify the partition to submit to."
        )


# Definition of project-related labels (classification)
@BondMSIBI.label
def completed(job):
    return job.doc.get("done")


def get_file(job, file_name):
    return os.path.abspath(os.path.join(job.ws, "..", "..", file_name))



@BondMSIBI.post(completed)
@BondMSIBI.operation(
    directives={"ngpu": 1, "executable": "python -u"}, name="optimize"
)
def optimize(job):
    from msibi import MSIBI, State, Bond
    import hoomd
    import os
    
    with job:
        job.doc["done"] = False
        if os.path.exists(job.fn("states")):
            dir_path = job.fn("states")
            os.system(f"rm -r {dir_path}")
        print("Setting up MSIBI optimizer...")
        opt = MSIBI(
            nlist=job.sp.nlist,
            integrator_method=hoomd.md.methods.ConstantVolume,
            method_kwargs={},
            thermostat=hoomd.md.methods.thermostats.MTTK,
            thermostat_kwargs={"tau": job.sp.thermostat_tau},
            dt=job.sp.dt,
            gsd_period=job.sp.n_steps // 500,
            nlist_exclusions=job.sp.nlist_exclusions,
        )

        print("Creating State objects...")
        single_chain_project = signac.get_project(job.sp.single_chain_path)
        for idx, state in enumerate(job.sp.states):
            print("state: ", state)
            single_chain_job = [
                j for j in single_chain_project.find_jobs(
                    filter={"lengths": 60,
                            "kT": state["kT"],
                            "remove_hydrogens": state["remove_hydrogens"]
                            }
                )
            ][0]
            gsd_file = single_chain_job.fn(
                f"cg-trajectory{single_chain_job.doc.runs - 1}.gsd")
            opt.add_state(
                State(
                    name=state["name"],
                    kT=single_chain_job.sp.kT,
                    traj_file=gsd_file,
                    n_frames=state["n_frames"],
                    alpha=state["alpha"],
                )
            )

        AA_bond = Bond(type1="A", type2="A", optimize=False, nbins=100)
        AA_bond.set_from_file(file_path="AA_bond.csv")
        opt.add_force(AA_bond)

        print("Creating Bond objects...")
        for bond in job.sp.bonds:
            _bond = Bond(
                type1=bond["type1"],
                type2=bond["type2"],
                optimize=True,
                nbins=job.sp.bonds_nbins,
                correction_form=job.sp.head_correction
            )
            _bond.set_quadratic(k4=bond["k4"], k3=bond["k3"], k2=bond["k2"],
                                x0=bond["x0"], x_min=bond["x_min"],
                                x_max=bond["x_max"])

            opt.add_force(_bond)
        print("Running Optimization...")

        opt.run_optimization(n_steps=job.sp.n_steps,
                             n_iterations=job.sp.n_iterations,
                             backup_trajectories=True)

        # save the optimized bonds to file
        for bond in opt.bonds:
            bond.save_to_file(job.fn(f"{bond.name}_bond.csv"))
            bond.plot_potentials(file_path=job.fn(f"{bond.name}_potential.png"))
            bond.plot_potential_history(file_path=job.fn(f"{bond.name}_potential_history.png"))

        # save plots to file
        for state in opt.states:
            for bond in opt.bonds:
                bond.plot_fit_scores(state=state, file_path=job.fn(f"{state.name}_{bond.name}_fitscore.png"))
                bond.plot_target_distribution(state=state, file_path=job.fn(f"{state.name}_{bond.name}_target_dist.png"))
                
                bond.plot_distribution_comparison(state=state, file_path=job.fn(f"{state.name}_{bond.name}_dist_comparison.png"))
                
        
        
        print("Optimization done")
        job.doc["done"] = True


if __name__ == "__main__":
    BondMSIBI(environment=Fry).main()
