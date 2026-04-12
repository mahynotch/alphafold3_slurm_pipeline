import json
import shutil
import subprocess
from itertools import product
from math import ceil
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Union

import numpy as np
import polars as pl
from tqdm import tqdm

from .config import Config
from .input_utils import (
    build_dimer,
    build_monomer,
    build_multimer,
    filter_dataframes,
    prepare_output_dir,
    read_file_as_df,
)
from .stat_utils import (
    collect_statistics,
    collect_statistics_exact,
    plot_confidence_boxplot,
)

config = Config()

MOLECULE_TYPES = {"protein", "ligand_ccd", "ligand_smiles", "dna", "rna"}


class BaseAlphafold3:
    def __init__(
        self,
        job_name: str,
        job_type: Literal["make_feature", "make_complex", "both"],
        destination: str,
        num_sample: int,
        time_each_protein: int,
        max_jobs: int,
        memory: int,
        num_cpu: int,
        gpu_type: Literal["a100", "v100"],
        flag_check: bool,
        flag_detailed: bool,
        flag_stat: bool,
        email: Optional[str] = None,
        overwrite: bool = False,
    ):
        self.job_name = job_name
        self.job_type = job_type
        self.destination = Path(destination)
        self.num_sample = num_sample
        self.time_each_protein = time_each_protein
        self.max_jobs = max_jobs
        self.memory = memory
        self.num_cpu = num_cpu
        self.gpu_type = gpu_type
        self.flag_check = flag_check
        self.flag_detailed = flag_detailed
        self.flag_stat = flag_stat
        self.email = email
        self.overwrite = overwrite

    def _cap_jobs(self, total_items: int) -> int:
        if total_items <= 0:
            return self.max_jobs
        return min(self.max_jobs, total_items)

    def _items_per_job(self, total_items: int) -> int:
        if total_items <= 0:
            return 1
        return max(1, ceil(total_items / self.max_jobs))

    def _compute_time(self, num_items: int) -> str:
        total_minutes = max(1, self.time_each_protein * max(1, num_items))
        hours, minutes = divmod(total_minutes, 60)
        return f"{hours:02}:{minutes:02}:00"

    def _get_inputs_dir(self) -> Path:
        return self.destination / f"inputs_{self.job_type}"

    def _get_job_dir(self, job_index: int) -> Path:
        return self._get_inputs_dir() / f"job-{job_index}"

    def _get_output_dir(self, name: str) -> Path:
        return self.destination / name

    def _get_feature_output_path(self, name: str) -> Path:
        return self._get_output_dir(name) / f"{name}_data.json"

    def _has_feature_output(self, name: str) -> bool:
        return self._get_feature_output_path(name).exists()

    def _has_structure_output(self, name: str) -> bool:
        output_dir = self._get_output_dir(name)
        return output_dir.exists() and any(path.suffix == ".cif" for path in output_dir.iterdir())

    def _prepare_inputs_dir(self) -> Path:
        return prepare_output_dir(self._get_inputs_dir(), overwrite=self.overwrite)

    def _write_job_input(self, job_index: int, name: str, input_json: str) -> Path:
        target_dir = self._get_job_dir(job_index)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{name}.json"
        target_path.write_text(input_json)
        return target_path

    def _stage_pending_inputs(self, pending_jobs: list[tuple[str, str]]) -> int:
        if not pending_jobs:
            print(f"No jobs remain to stage for {self.job_name}.")
            return 0

        self._prepare_inputs_dir()
        for index, (name, input_json) in enumerate(pending_jobs):
            self._write_job_input(index // self.items_per_job, name, input_json)
        return len(pending_jobs)

    def _read_feature_entry(self, feature_root: Path, name: str) -> dict | None:
        feature_path = feature_root / name / f"{name}_data.json"
        if not feature_path.exists():
            print(f"Feature file for {name} not found at {feature_path}, skipping...")
            return None

        with feature_path.open("r") as handle:
            data = json.load(handle)
        return data["sequences"][0]

    def _set_feature_chain_id(self, feature: dict, molecule_type: str, chain_id: str) -> dict:
        if molecule_type not in MOLECULE_TYPES:
            raise ValueError(f"Unsupported molecule type {molecule_type!r}")
        if molecule_type in {"ligand_ccd", "ligand_smiles"}:
            feature["ligand"]["id"] = chain_id
        else:
            feature[molecule_type]["id"] = chain_id
        return feature

    def _print_completion_status(
        self,
        names: list[str],
        predicate: Callable[[str], bool],
    ) -> None:
        status = np.zeros(len(names), dtype=int)
        for index, name in enumerate(names):
            if predicate(name):
                print(f"{name} done")
                status[index] = 1
            else:
                print(f"{name} not done")

        completed = int(np.sum(status))
        print(
            f"Total jobs: {len(names)}, Completed: {completed}, Failed: {len(names) - completed}"
        )
        if completed < len(names):
            print("Failed jobs:")
            for failed_name in np.array(names)[status == 0]:
                print(f"  {failed_name}")

    def _get_python_command(self) -> str:
        base_cmd = (
            f"run_alphafold --json_path=$json --model_dir={config.parameter} "
            f"--db_dir={config.db} --flash_attention_implementation=xla"
        )
        if self.job_type == "make_feature":
            return f"{base_cmd} --output_dir={self.destination} --norun_inference"
        if self.job_type == "make_complex":
            return (
                f"{base_cmd} --num_diffusion_samples {self.num_sample} "
                f"--output_dir={self.destination} --norun_data_pipeline"
            )
        return (
            f"{base_cmd} --num_diffusion_samples {self.num_sample} "
            f"--output_dir={self.destination}"
        )

    def print_script(self) -> Path:
        inputs_dir = self._get_inputs_dir()
        job_dirs = sorted(path for path in inputs_dir.iterdir() if path.is_dir())
        if not job_dirs:
            raise ValueError(f"No jobs found in {inputs_dir}")

        items_per_job = len(list(job_dirs[0].glob("*.json")))
        gpu_param = ""
        if self.job_type in {"make_complex", "both"}:
            gpu_param = f"#SBATCH --gres=gpu:1\n#SBATCH --constraint={self.gpu_type}"

        email_settings = ""
        if self.email:
            email_settings = f"#SBATCH --mail-type=ALL\n#SBATCH --mail-user={self.email}"

        (self.destination / "ibex_out").mkdir(parents=True, exist_ok=True)
        script_dir = self.destination / "script"
        script_dir.mkdir(parents=True, exist_ok=True)

        inputs_glob = f"{inputs_dir}/job-$SLURM_ARRAY_TASK_ID/*.json"
        script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-{len(job_dirs) - 1}
#SBATCH --job-name={self.job_name}
#SBATCH --output={self.destination / 'ibex_out' / '%x-%j.out'}
#SBATCH --time={self._compute_time(items_per_job)}
#SBATCH --mem={self.memory}G
#SBATCH --cpus-per-task={self.num_cpu}
{email_settings}
{gpu_param}
source ~/.bashrc
conda activate {config.env}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TF_FORCE_UNIFIED_MEMORY=1
export LA_FLAGS=\"--xla_gpu_enable_triton_gemm=false\"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_FLAGS=\"--xla_disable_hlo_passes=custom-kernel-fusion-rewriter\"

if [ -d {inputs_dir}/job-$SLURM_ARRAY_TASK_ID ]; then
    echo 'Directory exists, proceeding with the job...'
else
    echo 'Directory does not exist, exiting...'
    exit 1
fi

for json in {inputs_glob}; do
    echo $json
    echo {self.job_type}
    echo SOJ_indicator
    time {self._get_python_command()}
    echo $?
    echo EOJ_indicator
done
"""
        script_path = script_dir / f"run-{self.job_type}.slurm"
        script_path.write_text(script)
        return script_path

    def _submit_staged_jobs(self) -> None:
        inputs_dir = self._get_inputs_dir()
        has_jobs = inputs_dir.exists() and any(
            job_dir.is_dir() and any(job_dir.glob("*.json")) for job_dir in inputs_dir.iterdir()
        )
        if not has_jobs:
            print(f"No jobs remain to submit for {self.job_name}.")
            return

        script_path = self.print_script()
        self._sbatch_submit(script_path)

    def _sbatch_submit(self, script_path: Path) -> None:
        completed = subprocess.run(
            ["sbatch", str(script_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        print("Job submitted successfully")

    def detailed_check(self) -> None:
        ibex_out_dir = self.destination / "ibex_out"
        if not ibex_out_dir.exists():
            print(f"Output directory {ibex_out_dir} not found.")
            return

        for output_path in tqdm(sorted(ibex_out_dir.iterdir())):
            if not output_path.is_file():
                continue
            try:
                lines = output_path.read_text().splitlines()
            except OSError as error:
                print(f"Error checking {output_path.name}: {error}")
                continue

            current_file: Path | None = None
            for index, line in enumerate(lines):
                if "SOJ_indicator" in line and index >= 2:
                    current_file = Path(lines[index - 2].strip())
                    current_type = lines[index - 1].strip()
                    if current_type != self.job_type:
                        break
                elif "EOJ_indicator" in line and current_file is not None:
                    exit_code = lines[index - 1].strip() if index > 0 else "unknown"
                    if exit_code == "0":
                        print(f"{current_file} done")
                    else:
                        print(f"{current_file} failed, last lines:")
                        for offset in range(min(5, index - 1), 0, -1):
                            print(lines[index - 1 - offset])

                        output_dir = self.destination / current_file.stem
                        if output_dir.exists():
                            shutil.rmtree(output_dir, ignore_errors=True)


class Alphafold3PullDown(BaseAlphafold3):
    def __init__(
        self,
        job_name: str,
        job_type: Literal["make_feature", "make_complex", "both"],
        bait_type: Literal["protein", "ligand_ccd", "ligand_smiles", "dna", "rna"],
        bait_path: Union[str, list[str]],
        prey_type: Literal["protein", "ligand_ccd", "ligand_smiles", "dna", "rna"],
        prey_path: Union[str, list[str]],
        destination: str,
        feature_path: Optional[str] = None,
        num_sample: int = 5,
        time_each_protein: int = 120,
        max_jobs: int = 1500,
        memory: int = 64,
        num_cpu: int = 8,
        gpu_type: Literal["a100", "v100"] = "a100",
        flag_check: bool = False,
        flag_detailed: bool = False,
        flag_stat: bool = False,
        email: Optional[str] = None,
        overwrite: bool = False,
    ):
        super().__init__(
            job_name,
            job_type,
            destination,
            num_sample,
            time_each_protein,
            max_jobs,
            memory,
            num_cpu,
            gpu_type,
            flag_check,
            flag_detailed,
            flag_stat,
            email,
            overwrite,
        )
        if job_type in {"make_feature", "both"} and feature_path is not None:
            raise ValueError(f"feature_path should not be provided for {job_type}")

        self.bait = read_file_as_df(bait_path, bait_type)
        self.prey = read_file_as_df(prey_path, prey_type)
        self.feature_path = Path(feature_path) if feature_path else None
        self.bait_type = bait_type
        self.prey_type = prey_type

        total_combinations = len(self.bait) * len(self.prey)
        self.max_jobs = self._cap_jobs(total_combinations)
        self.items_per_job = self._items_per_job(total_combinations)

    def _combined_feature_inputs(self) -> pl.DataFrame:
        return pl.concat([self.bait, self.prey]).unique().drop_nulls()

    def _all_complex_names(self) -> list[str]:
        return [
            f"{bait_id}-{prey_id}"
            for bait_id, _, _ in self.bait.iter_rows()
            for prey_id, _, _ in self.prey.iter_rows()
        ]

    def make_protein_features_inputs(self) -> int:
        combined = self._combined_feature_inputs()
        pending_jobs: list[tuple[str, str]] = []
        for sequence_id, sequence, molecule_type in tqdm(
            combined.iter_rows(), total=len(combined)
        ):
            if self._has_feature_output(sequence_id):
                print(f"{sequence_id} already exists, skipping...")
                continue
            pending_jobs.append(
                (sequence_id, build_monomer(sequence_id, molecule_type, str(sequence)))
            )
        return self._stage_pending_inputs(pending_jobs)

    def make_complex_inputs(self) -> int:
        feature_root = self.feature_path or self.destination
        pending_jobs: list[tuple[str, str]] = []
        for bait_id, bait_seq, bait_type in tqdm(self.bait.iter_rows(), total=len(self.bait)):
            bait_feature = self._read_feature_entry(feature_root, bait_id)
            if bait_feature is None:
                continue

            for prey_id, prey_seq, prey_type in self.prey.iter_rows():
                prey_feature = self._read_feature_entry(feature_root, prey_id)
                if prey_feature is None:
                    continue

                name = f"{bait_id}-{prey_id}"
                if self._has_structure_output(name):
                    print(f"{name} already exists, skipping...")
                    continue

                prey_feature = self._set_feature_chain_id(prey_feature, prey_type, "B")
                input_json = json.loads(
                    build_dimer(name, bait_type, str(bait_seq), prey_type, str(prey_seq))
                )
                input_json["sequences"] = [bait_feature, prey_feature]
                pending_jobs.append((name, json.dumps(input_json)))

        return self._stage_pending_inputs(pending_jobs)

    def make_both_inputs(self) -> int:
        pending_jobs: list[tuple[str, str]] = []
        for bait_id, bait_seq, bait_type in self.bait.iter_rows():
            for prey_id, prey_seq, prey_type in self.prey.iter_rows():
                name = f"{bait_id}-{prey_id}"
                if self._has_structure_output(name):
                    print(f"{name} already exists, skipping...")
                    continue

                pending_jobs.append(
                    (name, build_dimer(name, bait_type, str(bait_seq), prey_type, str(prey_seq)))
                )

        return self._stage_pending_inputs(pending_jobs)

    def check_job(self) -> None:
        if self.job_type == "make_feature":
            names = list(self._combined_feature_inputs()["id"])
            self._print_completion_status(names, self._has_feature_output)
            return

        self._print_completion_status(self._all_complex_names(), self._has_structure_output)

    def check_stat(self) -> None:
        stat_dest = self.destination / "statistics"
        stat_dest.mkdir(parents=True, exist_ok=True)
        df = collect_statistics((list(self.bait["id"]), list(self.prey["id"])), self.destination)
        plot_confidence_boxplot(df, stat_dest / "statistics.png")
        df.write_csv(stat_dest / "statistics.csv")
        print(f"Statistics saved to {stat_dest}")

    def run(self) -> None:
        print(f"Running AlphaFold3 {self.job_type} job: {self.job_name}")
        if not any([self.flag_check, self.flag_detailed, self.flag_stat]):
            if self.job_type == "make_feature":
                staged_jobs = self.make_protein_features_inputs()
            elif self.job_type == "make_complex":
                staged_jobs = self.make_complex_inputs()
            else:
                staged_jobs = self.make_both_inputs()

            if staged_jobs:
                self._submit_staged_jobs()

        if self.flag_check:
            self.check_job()
        if self.flag_detailed:
            self.detailed_check()
        if self.flag_stat:
            self.check_stat()


class Alphafold3WrapperMonomer(BaseAlphafold3):
    def __init__(
        self,
        job_name: str,
        sequences: list[str],
        destination: str,
        num_sample: int = 5,
        time_each_protein: int = 120,
        max_jobs: int = 1500,
        memory: int = 64,
        num_cpu: int = 8,
        gpu_type: Literal["a100", "v100"] = "a100",
        flag_check: bool = False,
        flag_detailed: bool = False,
        flag_stat: bool = False,
        email: Optional[str] = None,
        overwrite: bool = False,
    ):
        super().__init__(
            job_name,
            "both",
            destination,
            num_sample,
            time_each_protein,
            max_jobs,
            memory,
            num_cpu,
            gpu_type,
            flag_check,
            flag_detailed,
            flag_stat,
            email,
            overwrite,
        )
        self.sequences = read_file_as_df(sequences, type_input="protein")
        self.max_jobs = self._cap_jobs(len(self.sequences))
        self.items_per_job = self._items_per_job(len(self.sequences))

    def make_protein_monomer_inputs(self) -> int:
        pending_jobs: list[tuple[str, str]] = []
        for sequence_id, sequence, molecule_type in self.sequences.iter_rows():
            if self._has_structure_output(sequence_id):
                print(f"{sequence_id} already exists, skipping...")
                continue
            pending_jobs.append(
                (sequence_id, build_monomer(sequence_id, molecule_type, str(sequence)))
            )
        return self._stage_pending_inputs(pending_jobs)

    def check_job(self) -> None:
        self._print_completion_status(list(self.sequences["id"]), self._has_structure_output)

    def check_stat(self) -> None:
        stat_dest = self.destination / "statistics"
        stat_dest.mkdir(parents=True, exist_ok=True)
        df = collect_statistics([list(self.sequences["id"])], self.destination)
        plot_confidence_boxplot(df, stat_dest / "statistics.png")
        df.write_csv(stat_dest / "statistics.csv")
        print(f"Statistics saved to {stat_dest}")

    def run(self) -> None:
        print(f"Running AlphaFold3 monomer job: {self.job_name}")
        if not any([self.flag_check, self.flag_detailed, self.flag_stat]):
            staged_jobs = self.make_protein_monomer_inputs()
            if staged_jobs:
                self._submit_staged_jobs()

        if self.flag_check:
            self.check_job()
        if self.flag_detailed:
            self.detailed_check()
        if self.flag_stat:
            self.check_stat()


class Alphafold3Multimer(BaseAlphafold3):
    def __init__(
        self,
        job_name: str,
        job_type: Literal["make_feature", "make_complex", "both"],
        input_types: list[Literal["protein", "ligand_ccd", "ligand_smiles", "dna", "rna"]],
        input_paths: list[Union[str, list[str]]],
        destination: str,
        feature_path: Optional[str] = None,
        exact: bool = False,
        num_sample: int = 5,
        time_each_protein: int = 120,
        max_jobs: int = 1500,
        memory: int = 64,
        num_cpu: int = 8,
        gpu_type: Literal["a100", "v100"] = "a100",
        flag_check: bool = False,
        flag_detailed: bool = False,
        flag_stat: bool = False,
        email: Optional[str] = None,
        overwrite: bool = False,
    ):
        super().__init__(
            job_name,
            job_type,
            destination,
            num_sample,
            time_each_protein,
            max_jobs,
            memory,
            num_cpu,
            gpu_type,
            flag_check,
            flag_detailed,
            flag_stat,
            email,
            overwrite,
        )
        if len(input_types) != len(input_paths):
            raise ValueError("input_types and input_paths must have the same length")
        if job_type in {"make_feature", "both"} and feature_path is not None:
            raise ValueError(f"feature_path should not be provided for {job_type}")

        self.sequence_lists = [
            read_file_as_df(path, input_type)
            for input_type, path in zip(input_types, input_paths)
        ]
        raw_lengths = [len(sequence_list) for sequence_list in self.sequence_lists]
        if exact and len(set(raw_lengths)) > 1:
            raise ValueError("Exact mode requires all input lists to have the same length")
        if exact:
            self.sequence_lists = filter_dataframes(self.sequence_lists)

        self.feature_path = Path(feature_path) if feature_path else None
        self.exact = exact
        self.type_list = input_types

        if self.job_type == "make_feature":
            total_items = len(pl.concat(self.sequence_lists).unique().drop_nulls())
        elif self.exact:
            total_items = len(self.sequence_lists[0])
        else:
            total_items = int(np.prod([len(seq_list) for seq_list in self.sequence_lists]))

        self.total_length = total_items
        self.max_jobs = self._cap_jobs(total_items)
        self.items_per_job = self._items_per_job(total_items)

    def _get_combinations(self) -> Iterable[tuple[tuple[str, str, str], ...]]:
        iterables = [sequence_list.iter_rows() for sequence_list in self.sequence_lists]
        if self.exact:
            return zip(*iterables)
        return product(*iterables)

    def _build_complex_name(self, combination: tuple[tuple[str, str, str], ...]) -> str:
        return "-".join(sequence_id for sequence_id, _, _ in combination)

    def _combined_feature_inputs(self) -> pl.DataFrame:
        return pl.concat(self.sequence_lists).unique().drop_nulls()

    def _all_complex_names(self) -> list[str]:
        return [self._build_complex_name(combination) for combination in self._get_combinations()]

    def make_protein_features_inputs(self) -> int:
        combined = self._combined_feature_inputs()
        pending_jobs: list[tuple[str, str]] = []
        for sequence_id, sequence, molecule_type in tqdm(
            combined.iter_rows(), total=len(combined)
        ):
            if self._has_feature_output(sequence_id):
                print(f"{sequence_id} already exists, skipping...")
                continue
            pending_jobs.append(
                (sequence_id, build_monomer(sequence_id, molecule_type, str(sequence)))
            )
        return self._stage_pending_inputs(pending_jobs)

    def make_complex_inputs(self) -> int:
        feature_root = self.feature_path or self.destination
        pending_jobs: list[tuple[str, str]] = []
        for combination in tqdm(self._get_combinations(), total=self.total_length):
            name = self._build_complex_name(combination)
            if self._has_structure_output(name):
                print(f"{name} already exists, skipping...")
                continue

            features: list[dict] = []
            missing_feature = False
            for chain_index, (sequence_id, _, molecule_type) in enumerate(combination):
                if chain_index >= 26:
                    raise ValueError("Too many sequences, should be less than or equal to 26")
                feature = self._read_feature_entry(feature_root, sequence_id)
                if feature is None:
                    missing_feature = True
                    break
                chain_id = chr(65 + chain_index)
                features.append(self._set_feature_chain_id(feature, molecule_type, chain_id))

            if missing_feature:
                continue

            input_json = json.loads(
                build_multimer(name, self.type_list, [""] * len(self.type_list), is_feature=True)
            )
            input_json["sequences"] = features
            pending_jobs.append((name, json.dumps(input_json)))

        return self._stage_pending_inputs(pending_jobs)

    def make_both_inputs(self) -> int:
        pending_jobs: list[tuple[str, str]] = []
        for combination in tqdm(self._get_combinations(), total=self.total_length):
            name = self._build_complex_name(combination)
            if self._has_structure_output(name):
                print(f"{name} already exists, skipping...")
                continue

            sequences = [str(sequence) for _, sequence, _ in combination]
            pending_jobs.append((name, build_multimer(name, self.type_list, sequences)))

        return self._stage_pending_inputs(pending_jobs)

    def check_job(self) -> None:
        if self.job_type == "make_feature":
            names = list(self._combined_feature_inputs()["id"])
            self._print_completion_status(names, self._has_feature_output)
            return
        self._print_completion_status(self._all_complex_names(), self._has_structure_output)

    def check_stat(self) -> None:
        stat_dest = self.destination / "statistics"
        stat_dest.mkdir(parents=True, exist_ok=True)
        id_lists = [list(sequence_list["id"]) for sequence_list in self.sequence_lists]
        if self.exact:
            df = collect_statistics_exact(id_lists, self.destination)
        else:
            df = collect_statistics(id_lists, self.destination)
        plot_confidence_boxplot(df, stat_dest / "statistics.png")
        df.write_csv(stat_dest / "statistics.csv")
        print(f"Statistics saved to {stat_dest}")

    def run(self) -> None:
        print(f"Running AlphaFold3 multimer {self.job_type} job: {self.job_name}")
        if not any([self.flag_check, self.flag_detailed, self.flag_stat]):
            if self.job_type == "make_feature":
                staged_jobs = self.make_protein_features_inputs()
            elif self.job_type == "make_complex":
                staged_jobs = self.make_complex_inputs()
            else:
                staged_jobs = self.make_both_inputs()

            if staged_jobs:
                self._submit_staged_jobs()

        if self.flag_check:
            self.check_job()
        if self.flag_detailed:
            self.detailed_check()
        if self.flag_stat:
            self.check_stat()
