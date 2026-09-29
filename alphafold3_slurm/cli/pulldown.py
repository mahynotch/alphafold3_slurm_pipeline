"""af3pulldown: AlphaPulldown-style bait x prey screens."""

import argparse

from ..wrapper import Alphafold3PullDown
from ._common import (
    MOLECULE_TYPE_HELP,
    add_check_arguments,
    add_slurm_arguments,
    add_stage_arguments,
    resolve_job_type,
)


def parsing(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Takes one or more FASTA/CSV/TSV files of baits and preys, and submits "
            "a Slurm job array to predict the structure of each bait-prey pair."
        )
    )
    parser.add_argument(
        "--job_name",
        help="Name of the job. This is the name shown in squeue.",
        default="AF3_pulldown",
    )
    parser.add_argument("--bait_type", help=f"Bait type: {MOLECULE_TYPE_HELP}.", required=True)
    parser.add_argument(
        "--bait_input",
        help="FASTA/CSV/TSV file(s) with bait sequences.",
        nargs="+",
        required=True,
    )
    parser.add_argument("--prey_type", help=f"Prey type: {MOLECULE_TYPE_HELP}.", required=True)
    parser.add_argument(
        "--prey_input",
        help="FASTA/CSV/TSV file(s) with prey sequences.",
        nargs="+",
        required=True,
    )
    add_stage_arguments(parser)
    add_slurm_arguments(parser, default_time=300)
    add_check_arguments(parser)
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> None:
    args = parsing(argv)
    job_type = resolve_job_type(args)
    Alphafold3PullDown(
        args.job_name,
        job_type,
        args.bait_type,
        args.bait_input,
        args.prey_type,
        args.prey_input,
        destination=args.destination,
        feature_path=args.feature_path,
        time_each_protein=args.time,
        memory=args.mem,
        email=args.mail,
        max_jobs=args.max_jobs,
        flag_check=args.check_only,
        flag_detailed=args.check_only_exact,
        flag_stat=args.check_stat,
        gpu_type=args.gpu_type,
        overwrite=args.overwrite,
    ).run()


if __name__ == "__main__":
    main()
