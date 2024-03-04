"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help

"""

import os

import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment


class PairMSIBI(FlowProject):
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
@PairMSIBI.label
def completed(job):
    return job.doc.get("done")


def get_file(job, file_name):
    return os.path.abspath(os.path.join(job.ws, "..", "..", file_name))



@PairMSIBI.post(completed)
@PairMSIBI.operation(
    directives={"ngpu": 1, "executable": "python -u"}, name="optimize"
)
def optimize(job):
    from msibi import MSIBI, State, Bond, Angle, Pair
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
        bulk_project = signac.get_project(job.sp.bulk_path)
        for idx, state in enumerate(job.sp.states):
            print("state: ", state)
            bulk_job = bulk_project.open_job(id=state["job_id"])
            gsd_file = bulk_job.fn("last-npt-cg.gsd")
            opt.add_state(
                State(
                    name=state["name"],
                    kT=state["kT"],
                    traj_file=gsd_file,
                    n_frames=state["n_frames"],
                    alpha=state["alpha"],
                    exclude_bonded=state["exclude_bonded"],
                )
            )

        print("Creating Bond objects...")
        bond_project = signac.get_project(job.sp.bond_project_path)
        bond_job = bond_project.open_job(id=job.sp.bond_job_id)
        for bond in job.sp.bonds:
            _bond = Bond(
                type1=bond["type1"],
                type2=bond["type2"],
                optimize=False,
                nbins=job.sp.bonds_nbins,
            )
            _bond.set_from_file(file_apth=bond_job.fn(bond["file_path"]))
            opt.add_force(_bond)

        print("Creating Angle objects...")
        for angle in job.sp.angles:
            _angle = Angle(
                type1=angle["type1"],
                type2=angle["type2"],
                type3=angle["type3"],
                optimize=False,
                nbins=job.sp.angles_nbins,
            )
            _angle.set_from_file(file_apth=angle["file_path"])
            opt.add_force(_angle)

        print("Creating Pair objects...")
        for pair in job.sp.pairs:
            _pair = Pair(
                type1=pair["type1"],
                type2=pair["type2"],
                r_cut=pair["r_cut"],
                optimize=True,
                nbins=job.sp.pairs_nbins,
            )
            _pair.set_lj(epsilon=pair["epsilon"], sigma=pair["sigma"], r_cut=pair["r_cut"])
            opt.add_force(_pair)

        print("Running Optimization...")
        opt.run_optimization(n_steps=job.sp.n_steps,
                             n_iterations=job.sp.n_iterations,
                             backup_trajectories=True)

        # save the optimized angles to file
        for pair in opt.pairs:
            pair.save_to_file(job.fn(f"{pair.name}_pair.csv"))
            pair.plot_potentials(file_path=job.fn(f"{pair.name}_potential.png"))
            pair.plot_potential_history(file_path=job.fn(f"{pair.name}_potential_history.png"))

        # save plots to file
        for state in opt.states:
            for pair in opt.pairs:
                pair.plot_fit_scores(state=state, file_path=job.fn(f"{state.name}_{pair.name}_fitscore.png"))
                pair.plot_target_distribution(state=state, file_path=job.fn(f"{state.name}_{pair.name}_target_dist.png"))

                pair.plot_distribution_comparison(state=state, file_path=job.fn(f"{state.name}_{pair.name}_dist_comparison.png"))


        print("Optimization done")
        job.doc["done"] = True


if __name__ == "__main__":
    PairMSIBI(environment=Fry).main()
