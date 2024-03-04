"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help

"""

import os

import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment


class DihedralMSIBI(FlowProject):
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
@DihedralMSIBI.label
def completed(job):
    return job.doc.get("done")


def get_file(job, file_name):
    return os.path.abspath(os.path.join(job.ws, "..", "..", file_name))


@DihedralMSIBI.post(completed)
@DihedralMSIBI.operation(
    directives={"ngpu": 1, "executable": "python -u"}, name="optimize"
)
def optimize(job):
    from msibi import MSIBI, State, Bond, Angle, Pair, Dihedral
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
                optimize=False,
                nbins=job.sp.pairs_nbins,
            )
            _pair.set_from_file(file_apth=pair["file_path"])
            opt.add_force(_pair)

        print("Creating Dihedral objects...")
        for dihedral in job.sp.dihedrals:
            _dihedral = Dihedral(
                type1=dihedral["type1"],
                type2=dihedral["type2"],
                type3=dihedral["type3"],
                type4=dihedral["type4"],
                optimize=True,
                nbins=job.sp.dihedrals_nbins,
            )
            _dihedral.set_quartic(
                x0=dihedral["x0"],
                x_min=dihedral["x_min"],
                x_max=dihedral["x_max"],
                k2=dihedral["k2"],
                k3=dihedral["k3"],
                k4=dihedral["k4"],
            )
            opt.add_force(_dihedral)

        print("Running Optimization...")
        opt.run_optimization(n_steps=job.sp.n_steps,
                             n_iterations=job.sp.n_iterations,
                             backup_trajectories=True)

        # save the optimized angles to file
        for dihedral in opt.dihedral:
            dihedral.save_to_file(job.fn(f"{dihedral.name}_dihedral.csv"))
            dihedral.plot_potentials(
                file_path=job.fn(f"{dihedral.name}_potential.png"))
            dihedral.plot_potential_history(
                file_path=job.fn(f"{dihedral.name}_potential_history.png"))

        # save plots to file
        for state in opt.states:
            for dihedral in opt.dihedrals:
                dihedral.plot_fit_scores(state=state, file_path=job.fn(
                    f"{state.name}_{dihedral.name}_fitscore.png"))
                dihedral.plot_target_distribution(state=state, file_path=job.fn(
                    f"{state.name}_{dihedral.name}_target_dist.png"))

                dihedral.plot_distribution_comparison(state=state,
                                                      file_path=job.fn(
                                                          f"{state.name}_{dihedral.name}_dist_comparison.png"))

        print("Optimization done")
        job.doc["done"] = True


if __name__ == "__main__":
    DihedralMSIBI(environment=Fry).main()
