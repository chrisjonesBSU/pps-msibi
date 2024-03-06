"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help

"""

import os

import signac
from flow import FlowProject
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
            "--partition", default="batch," "v100",
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
            gsd_period=job.sp.n_steps[0] // 500,
            nlist_exclusions=job.sp.nlist_exclusions,
        )

        print("Creating State objects...")
        single_chain_project = signac.get_project(job.sp.single_chain_path)
        single_chain_job = single_chain_project.open_job(
            id=job.sp.single_chain_job_id
        )
        job.doc.target_state_path = single_chain_job.path
        for idx, state in enumerate(job.sp.states):
            print("state: ", state)
            gsd_file = single_chain_job.fn(state["cg_file_name"])
            state = State(
                name=state["name"],
                kT=single_chain_job.sp.kT,
                traj_file=gsd_file,
                n_frames=state["n_frames"],
                alpha=job.sp.state_alphas[0],
            )
            opt.add_state(state)

        AA_bond = Bond(
                type1=job.sp.bonds["type1"],
                type2=job.sp.bonds["type2"],
                optimize=True,
                nbins=job.sp.bonds_nbins
        )
        AA_bond.set_quadratic(
            x0=job.sp.bonds["x0"],
            x_min=job.sp.bonds["x_min"],
            x_max=job.sp.bonds["x_max"],
            k2=job.sp.bonds["k2"],
            k3=job.sp.bonds["k3"],
            k4=job.sp.bonds["k4"],
        )
        opt.add_force(AA_bond)

        print("Running Optimization...")

        for n_iterations, n_steps, alpha in zip(
                job.sp.n_iterations, job.sp.n_steps, job.sp.state_alphas
        ):
            state.alpha = alpha
            opt.run_optimization(
                    n_steps=n_steps,
                    n_iterations=n_iterations,
                    backup_trajectories=True
            )
            AA_bond.smooth_potential()

        # save the optimized bonds to file
        AA_bond.save_potential(job.fn(f"{AA_bond.name}_bond.csv"))
        AA_bond.save_potential_history(
            job.fn(f"{AA_bond.name}_potential_history.npy")
        )
        AA_bond.plot_potentials(
            file_path=job.fn(f"{AA_bond.name}_potential.png")
        )
        AA_bond.plot_potential_history(
            file_path=job.fn(f"{AA_bond.name}_potential_history.png")
        )

        # save plots to file
        for state in opt.states:
            AA_bond.save_state_data(
                    state=state,
                    file_path=job.fn(
                        f"state_{state.name}_bond_{AA_bond.name}_data.npz"
                    )
            )
            AA_bond.plot_fit_scores(
                    state=state,
                    file_path=job.fn(
                        f"{state.name}_{AA_bond.name}_fitscore.png"
                    )
            )
            AA_bond.plot_target_distribution(
                    state=state,
                    file_path=job.fn(
                        f"{state.name}_{AA_bond.name}_target_dist.png"
                    )
            )
            AA_bond.plot_distribution_comparison(
                    state=state,
                    file_path=job.fn(
                        f"{state.name}_{AA_bond.name}_dist_comparison.png"
                    )
            )

        print("Optimization done")
        job.doc["done"] = True


if __name__ == "__main__":
    BondMSIBI(environment=Fry).main()
