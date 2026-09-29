"""af3oligomer: combinations of several input lists (A x B x C ...)."""

import argparse

from ..wrapper import Alphafold3Multimer
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
            "Takes one or more FASTA/CSV/TSV files and submits a Slurm job array "
            "to predict all requested multimer combinations."
        )
    )
    parser.add_argument(
        "--job_name",
        help="Name of the job. This is the name shown in squeue.",
        default="AF3_oligomer",
    )
    parser.add_argument(
        "--input_type",
        help=f"Input types in order: {MOLECULE_TYPE_HELP}.",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--input",
        help="FASTA/CSV/TSV file(s) with sequences. Order must match --input_type.",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--exact",
        help="Pair the nth record from each input instead of taking the Cartesian product.",
        action="store_true",
    )
    add_stage_arguments(parser)
    add_slurm_arguments(parser, default_time=600)
    add_check_arguments(parser)
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> None:
    args = parsing(argv)
    job_type = resolve_job_type(args)
    Alphafold3Multimer(
        args.job_name,
        job_type,
        args.input_type,
        args.input,
        destination=args.destination,
        feature_path=args.feature_path,
        exact=args.exact,
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
