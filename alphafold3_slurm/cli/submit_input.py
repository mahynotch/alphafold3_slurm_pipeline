"""submit_input: submit ready-made AlphaFold3 JSON inputs, one Slurm job each."""

import argparse
import shlex
import tempfile
from pathlib import Path

from ..config import get_config
from ..runtime import GPU_TYPES, build_af3_command, runtime_exports, sbatch_submit

SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name={job_name}
#SBATCH --output={output}/ibex_out/%x-%j.out
#SBATCH --time={time}
#SBATCH --mem={mem}G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --constraint={gpu_type}

{exports}

time {command}
"""


def parsing(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit one JSON input or a directory of JSON inputs to Slurm."
    )
    parser.add_argument(
        "--input",
        help="JSON file or directory containing JSON input files.",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Directory for AlphaFold outputs and Slurm logs.",
        required=True,
    )
    parser.add_argument("--time", help="Minutes to allocate per job.", type=int, default=300)
    parser.add_argument("--mem", help="Memory in GB per job.", type=int, default=64)
    parser.add_argument("--cpus", help="CPUs per job.", type=int, default=8)
    parser.add_argument(
        "--gpu_type", help="GPU type to request.", choices=GPU_TYPES, default="a100"
    )
    return parser.parse_args(args)


def time_conversion(time: int) -> str:
    hours = time // 60
    minutes = time % 60
    return f"{hours:02d}:{minutes:02d}:00"


def _render_script(
    input_path: Path,
    output_path: Path,
    time_minutes: int,
    memory_gb: int,
    gpu_type: str = "a100",
    cpus: int = 8,
) -> str:
    command = build_af3_command(
        get_config(),
        shlex.quote(str(input_path)),
        output_path,
        gpu_type=gpu_type,
        num_cpu=cpus,
        compilation_cache_dir=output_path / ".jax_cache",
    )
    return SCRIPT_TEMPLATE.format(
        job_name=f"AF3_{input_path.stem}",
        output=output_path,
        time=time_conversion(time_minutes),
        mem=memory_gb,
        cpus=cpus,
        gpu_type=gpu_type,
        exports=runtime_exports(gpu_type),
        command=command,
    )


def _submit_script(script_contents: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".slurm", delete=False) as handle:
        handle.write(script_contents)
        script_path = Path(handle.name)

    try:
        stdout = sbatch_submit(script_path)
        if stdout:
            print(stdout)
    finally:
        script_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    args = parsing(argv)
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    (output_path / "ibex_out").mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        json_paths = sorted(path for path in input_path.iterdir() if path.suffix == ".json")
    else:
        json_paths = [input_path]

    for json_path in json_paths:
        print(f"Submitting {json_path}")
        _submit_script(
            _render_script(json_path, output_path, args.time, args.mem, args.gpu_type, args.cpus)
        )


if __name__ == "__main__":
    main()
