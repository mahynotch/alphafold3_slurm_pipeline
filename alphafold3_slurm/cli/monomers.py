"""make_monomers: one prediction per input protein sequence."""

import argparse

from ..wrapper import Alphafold3WrapperMonomer
from ._common import add_check_arguments, add_slurm_arguments


def parsing(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Takes one or more FASTA/CSV/TSV files with amino acid sequences and "
            "submits a Slurm job array to predict each sequence."
        )
    )
    parser.add_argument("--job_name", help="Name of the job shown in squeue.", required=True)
    parser.add_argument(
        "--input",
        help="FASTA/CSV/TSV file(s) with monomer sequences.",
        nargs="+",
        required=True,
    )
    add_slurm_arguments(parser, default_time=300)
    add_check_arguments(parser)
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> None:
    args = parsing(argv)
    Alphafold3WrapperMonomer(
        args.job_name,
        args.input,
        destination=args.destination,
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
