"""Define the project's workflow logic and operation functions.

Execute this script directly from the command line, to view your project's
status, execute operations and submit them to a cluster. See also:

    $ python src/project.py --help

"""

import signac
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
import os


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


@directives(executable="python -u")
@directives(ngpu=1)
@BondMSIBI.operation
@BondMSIBI.post(completed)
def optimize(job):
    from msibi import MSIBI, State, Bond
    import logging

    with job:
        job.doc["done"] = False

        print("Setting up MSIBI optimizer...")
        opt = MSIBI(
            nlist=job.sp.nlist,
            integrator_method=job.sp.integrator,
            method_kwargs={},
            thermostat="MTTK",
            thermostat_kwargs={"tau": job.sp.thermostat_tau},
            dt=job.sp.dt,
            gsd_period=job.sp.n_steps // 500,
            r_cut=job.sp.r_cut,
            nlist_exclusions=job.sp.nlist_exclusions,
        )

        print("Creating State objects...")
        single_chain_project = signac.get_project(job.sp.single_chain_path)
        for idx, state in enumerate(job.sp.states):
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
                    max_frames=state["n_frames"],
                    alpha=state["alpha"],
                )
            )


        print("Creating Bond objects...")
        for bond in job.sp.bonds:
            _bond = Bond(
                type1=bond["type1"],
                type2=bond["type2"],
                optimize=True,
                nbins=job.sp.bonds_nbins,
                head_correction_form=job.sp.head_correction
            )
            _bond.set_quadratic(k4=bond["k4"], k3=bond["k3"], k2=bond["k2"],
                                x0=bond["x0"], x_min=bond["x_min"],
                                x_max=bond["x_max"])

            opt.add_bond(_bond)

        opt.run_optimization(n_steps=job.sp.n_steps,
                             n_iterations=job.sp.n_iterations,
                             backup_trajectories=True)

        # save the optimized bonds to file
        for bond in opt.bonds:
            bond.save_to_file(job.fn(f"{bond.name}.csv"))

        job.doc["done"] = True


if __name__ == "__main__":
    BondMSIBI().main()
