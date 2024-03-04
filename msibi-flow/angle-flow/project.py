"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help

"""

import os

import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment


class AngleMSIBI(FlowProject):
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
@AngleMSIBI.label
def completed(job):
    return job.doc.get("done")


def get_file(job, file_name):
    return os.path.abspath(os.path.join(job.ws, "..", "..", file_name))



@AngleMSIBI.post(completed)
@AngleMSIBI.operation(
    directives={"ngpu": 1, "executable": "python -u"}, name="optimize"
)
def optimize(job):
    from msibi import MSIBI, State, Bond, Angle
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
                optimize=True,
                nbins=job.sp.angles_nbins,
            )
            _angle.set_quadratic(
                x0=angle["x0"],
                x_min=angle["x_min"],
                x_max=angle["x_max"],
                k2=angle["k2"],
                k3=angle["k3"],
                k4=angle["k4"],
            )
            opt.add_force(_angle)

        print("Running Optimization...")
        opt.run_optimization(n_steps=job.sp.n_steps,
                             n_iterations=job.sp.n_iterations,
                             backup_trajectories=True)

        # save the optimized angles to file
        for angle in opt.angles:
            angle.save_to_file(job.fn(f"{angle.name}_angle.csv"))
            angle.plot_potentials(file_path=job.fn(f"{angle.name}_potential.png"))
            angle.plot_potential_history(file_path=job.fn(f"{angle.name}_potential_history.png"))

        # save plots to file
        for state in opt.states:
            for angle in opt.angles:
                angle.plot_fit_scores(state=state, file_path=job.fn(f"{state.name}_{angle.name}_fitscore.png"))
                angle.plot_target_distribution(state=state, file_path=job.fn(f"{state.name}_{angle.name}_target_dist.png"))

                angle.plot_distribution_comparison(state=state, file_path=job.fn(f"{state.name}_{angle.name}_dist_comparison.png"))


        print("Optimization done")
        job.doc["done"] = True


if __name__ == "__main__":
    AngleMSIBI(environment=Fry).main()
