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
    import numpy as np

    with job:
        print("Starting MSIBI Optimization for job:")
        print(job.id)
        job.doc["done"] = False
        if os.path.exists(job.fn("states")):
            dir_path = job.fn("states")
            os.system(f"rm -r {dir_path}")
        print("Setting up MSIBI optimizer...")
        opt = MSIBI(
            nlist=hoomd.md.nlist.Cell,
            integrator_method=hoomd.md.methods.ConstantVolume,
            method_kwargs={},
            thermostat=hoomd.md.methods.thermostats.MTTK,
            thermostat_kwargs={"tau": job.sp.thermostat_tau},
            dt=job.sp.dt,
            gsd_period=job.sp.n_steps[0] // 200,
            nlist_exclusions=job.sp.nlist_exclusions,
        )

        print("Creating State objects...")
        for state in job.sp.states:
            print("State: ", state)
            target_project = signac.get_project(state["target_project"])
            target_state_job = target_project.open_job(
                    id=state["target_job_id"]
            )
            print("Target Job:", target_state_job)
            gsd_file = target_state_job.fn(state["cg_file_name"])
            print("Target gsd: ", gsd_file)
            print()
            if state["name"] == "Ordered":
                t_scale = job.sp.T_scale
            else:
                t_scale = 1
            state = State(
                name=state["name"],
                kT=target_state_job.sp.kT * t_scale,
                traj_file=gsd_file,
                n_frames=state["n_frames"],
                alpha=1.0,
            )
            opt.add_state(state)

        print("Creating Angle objects...")
        angle_project = signac.get_project(job.sp.angle_project_path)
        angle_job = angle_project.open_job(id=job.sp.angle_job_id)

        AAA_angle = Angle(
            type1=job.sp.angles["type1"],
            type2=job.sp.angles["type2"],
            type3=job.sp.angles["type3"],
            optimize=False,
            nbins=angle_job.sp.angles_nbins,
        )
        AAA_angle.set_from_file(
                file_path=angle_job.fn(job.sp.angles["file_path"])
        )
        opt.add_force(AAA_angle)
        # Add harmonic bond
        print("Creating Bond objects...")
        AA_bond = Bond(
            type1="A",
            type2="A",
            optimize=False,
            nbins=100,
        )
        AA_bond.set_quadratic(
                x0=angle_job.sp.bond_l0,
                k2=angle_job.sp.bond_k / 2,
                k4=0,
                k3=0,
                x_min=0,
                x_max=3.0,
        )
        opt.add_force(AA_bond)

        print("Creaing Pair objects...")
        AA_pair = Pair(
                type1=job.sp.pairs["type1"],
                type2=job.sp.pairs["type2"],
                r_cut=job.sp.r_cut,
                nbins=job.sp.pairs_nbins,
                exclude_bonded=True,
                optimize=True
        )
        AA_pair.set_lj(
                epsilon=job.sp.epsilon,
                sigma=job.sp.sigma,
                r_min=0.1,
                r_cut=job.sp.r_cut
        )
        AA_pair.smoothing_window = 5
        opt.add_force(AA_pair)

        print("Running Optimization...")
        for n_iterations, n_steps, alphas in zip(
                job.sp.n_iterations,
                job.sp.n_steps,
                job.sp.state_alphas
        ):
            print("ALPHAS")
            print(alphas)
            for idx, state in enumerate(opt.states):
                state.alpha = alphas[idx]
            opt.run_optimization(
                    n_steps=n_steps,
                    n_iterations=n_iterations,
                    backup_trajectories=True
            )
            AA_pair.smooth_potential()
            AA_pair.save_potential(job.fn(f"pair_pot.csv"))

        # save the optimized pairs to file
        AA_pair.save_potential(job.fn(f"{AA_pair.name}_pair.csv"))
        AA_pair.save_potential_history(
            job.fn(f"{AA_pair.name}_potential_history.npy")
    )
        AA_pair.plot_potentials(
            file_path=job.fn(f"{AA_pair.name}_potential.png"),
            xlim=(1, job.sp.r_cut),
            ylim=(-3,50)
    )
        AA_pair.plot_potential_history(
            file_path=job.fn(f"{AA_pair.name}_potential_history.png"),
            xlim=(1, job.sp.r_cut),
            ylim=(-3,50)
    )

        # save plots to file
        for state in opt.states:
            AA_pair.save_state_data(
                    state=state,
                    file_path=job.fn(
                        f"state_{state.name}_pair_{AA_pair.name}_data.npz"
                    )
            )
            AA_pair.plot_fit_scores(
                    state=state,
                    file_path=job.fn(
                        f"{state.name}_{AA_pair.name}_fitscore.png"
                    )
            )
            AA_pair.plot_target_distribution(
                    state=state,
                    file_path=job.fn(
                        f"{state.name}_{AA_pair.name}_target_dist.png"
                    )
            )
            AA_pair.plot_distribution_comparison(
                    state=state,
                    file_path=job.fn(
                        f"{state.name}_{AA_pair.name}_dist_comparison.png"
                    )
            )

        print("Optimization done")
        job.doc["done"] = True


if __name__ == "__main__":
    PairMSIBI(environment=Fry).main()
