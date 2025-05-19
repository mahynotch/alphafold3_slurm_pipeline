from .input_utils import (
    build_monomer,
    read_file_as_df,
    build_dimer,
    check_exist,
    build_multimer,
    filter_dataframes
)
from .stat_utils import collect_statistics, plot_confidence_boxplot, collect_statistics_exact
from .config import Config
from typing import Literal, List, Optional, Union
from itertools import product
import json
import polars as pl
import numpy as np
import os
import shutil
from tqdm import tqdm
from pathlib import Path
import traceback

config = Config()


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
    ):
        self.job_name = job_name
        self.job_type = job_type
        self.destination = destination
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

    def _compute_time(self, num_items: int) -> str:
        """Calculate walltime based on number of items and job parameters."""
        total_minutes = self.time_each_protein * np.ceil(num_items / self.max_jobs)
        hours, minutes = divmod(int(total_minutes), 60)
        seconds = int((total_minutes - int(total_minutes)) * 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    
    def _get_inputs_dir(self) -> str:
        """Get the directory path for inputs of the current job type."""
        return os.path.join(self.destination, f"inputs_{self.job_type}")
    
    def _get_job_dir(self, job_index: int) -> str:
        """Get the directory path for a specific job index."""
        return os.path.join(self._get_inputs_dir(), f"job-{job_index}")
    
    def _get_python_command(self) -> str:
        """Get the Python command based on job type."""
        model_dir = config["parameter"]
        db_dir = config["db"]
        
        base_cmd = f"run_alphafold --json_path=$json --model_dir={model_dir} --db_dir={db_dir}"
        
        if self.job_type == "make_feature":
            return f"{base_cmd} --output_dir={self.destination} --norun_inference"
        elif self.job_type == "make_complex":
            return f"{base_cmd} --num_diffusion_samples {self.num_sample} --output_dir={self.destination} --norun_data_pipeline"
        else:  # both
            return f"{base_cmd} --num_diffusion_samples {self.num_sample} --output_dir={self.destination}"
    
    def print_script(self) -> str:
        """Generate the SLURM script and return the path to the script file."""
        inputs_dir = self._get_inputs_dir()
        job_num = len(os.listdir(inputs_dir)) if os.path.exists(inputs_dir) else 0
        
        if job_num == 0:
            raise ValueError(f"No jobs found in {inputs_dir}")
        
        # Determine first job's content for timing calculation
        first_job_dir = os.path.join(inputs_dir, "job-0")
        items_per_job = len(os.listdir(first_job_dir)) if os.path.exists(first_job_dir) else 1
        
        # Set GPU parameters if needed
        gpu_param = ""
        if self.job_type in ["make_complex", "both"]:
            gpu_param = f"#SBATCH --gres=gpu:1\n#SBATCH --constraint={self.gpu_type}"
        
        # Email notification settings
        email_settings = ""
        if self.email:
            email_settings = f"#SBATCH --mail-type=ALL\n#SBATCH --mail-user={self.email}"
        
        script = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH --array=0-{job_num - 1}
#SBATCH --job-name={self.job_name}
#SBATCH --output={os.path.join(self.destination, 'ibex_out', '%x-%j.out')}
#SBATCH --time={self._compute_time(items_per_job)}
#SBATCH --mem={self.memory}G
#SBATCH --cpus-per-task={self.num_cpu}
{email_settings}
{gpu_param}
source ~/.bashrc
conda activate {config['env']}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TF_FORCE_UNIFIED_MEMORY=1
export LA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

if [ -d {os.path.join(self.destination, f'inputs_{self.job_type}', 'job-$SLURM_ARRAY_TASK_ID')} ]; then
    echo 'Directory exists, proceeding with the job...'
else
    echo 'Directory does not exist, exiting...'
    exit 1
fi

for json in {os.path.join(self.destination, f'inputs_{self.job_type}', 'job-$SLURM_ARRAY_TASK_ID', '*.json')}; do
    echo $json
    echo {self.job_type}
    echo SOJ_indicator
    time {self._get_python_command()}
    echo $?
    echo EOJ_indicator
done
"""
        
        # Create script directory and save script
        script_dir = os.path.join(self.destination, "script")
        os.makedirs(script_dir, exist_ok=True)
        
        script_path = os.path.join(script_dir, f"run-{self.job_type}.slurm")
        with open(script_path, "w") as f:
            f.write(script)
            
        return script_path

    def _sbatch_submit(self, script_path: str) -> None:
        """Submit a job using sbatch."""
        try:
            output = os.popen(f"sbatch {script_path}").read()
            print(output)
            print("Job submitted successfully")
        except Exception as e:
            print(f"Error submitting job: {e}")

    def detailed_check(self) -> None:
        """Perform detailed check of job outputs to identify failures."""
        ibex_out_dir = os.path.join(self.destination, "ibex_out")
        if not os.path.exists(ibex_out_dir):
            print(f"Output directory {ibex_out_dir} not found.")
            return
            
        for file_name in tqdm(os.listdir(ibex_out_dir)):
            try:
                with open(os.path.join(ibex_out_dir, file_name), "r") as f:
                    lines = f.readlines()
                    current_file = None
                    
                    for index, line in enumerate(lines):
                        if "SOJ_indicator" in line and index >= 2:
                            current_file = lines[index - 2].strip()
                            current_type = lines[index - 1].strip()
                            if current_type != self.job_type:
                                break
                                
                        elif "EOJ_indicator" in line and current_file:
                            exit_code = lines[index - 1].strip() if index > 0 else "unknown"
                            if exit_code == "0":
                                print(f"{current_file} done")
                            else:
                                print(f"{current_file} failed, last lines:")
                                for i in range(min(5, index-1), 0, -1):
                                    print(lines[index - 1 - i])
                                    
                                # Clean up failed output if it exists
                                current_file_name = os.path.splitext(current_file)[0]
                                output_dir = os.path.join(self.destination, current_file_name)
                                if os.path.exists(output_dir):
                                    shutil.rmtree(output_dir, ignore_errors=True)
            except Exception as e:
                print(f"Error checking {file_name}: {e}")


class Alphafold3PullDown(BaseAlphafold3):
    """
    Class to run AlphaFold3 for protein-protein or protein-ligand interactions.
    Reminder: You must run with make features before running make complex.
    """

    def __init__(
        self,
        job_name: str,
        job_type: Literal["make_feature", "make_complex", "both"],
        bait_type: Literal["protein", "ligand_ccd", "ligand_smiles", "dna", "rna"],
        bait_path: Union[str, List[str]],
        prey_type: Literal["protein", "ligand_ccd", "ligand_smiles", "dna", "rna"],
        prey_path: Union[str, List[str]],
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
        )
        
        # Validate feature path based on job type
        if job_type == "make_complex" and not feature_path:
            raise ValueError("feature_path is required for make_complex")
        if job_type == "make_feature" and feature_path:
            raise ValueError("feature_path should not be provided for make_feature")
            
        # Load input sequences
        self.bait = read_file_as_df(bait_path, bait_type)
        self.prey = read_file_as_df(prey_path, prey_type)
        self.feature_path = feature_path
        self.bait_type = bait_type
        self.prey_type = prey_type
        
        # Determine job distribution
        total_combinations = len(self.bait) * len(self.prey)
        self.max_jobs = min(self.max_jobs, total_combinations) if total_combinations > 0 else self.max_jobs
        self.protein_per_job = max(1, total_combinations // self.max_jobs)

    def make_protein_features_inputs(self):
        """Generate input files for feature computation."""
        check_exist(self._get_inputs_dir())
        
        # Combine and deduplicate sequences
        combined = pl.concat([self.bait, self.prey]).unique().drop_nulls()
        self.combined = combined
        
        for index, (id, seq, type) in tqdm(enumerate(combined.iter_rows())):
            if os.path.exists(os.path.join(self.destination, id)):
                print(f"{id} already exists, skipping...")
                continue
                
            job_index = index // self.protein_per_job
            target_dir = self._get_job_dir(job_index)
            os.makedirs(target_dir, exist_ok=True)
            
            input_json = build_monomer(id, type, str(seq))
            with open(os.path.join(target_dir, f"{id}.json"), "w") as f:
                f.write(input_json)

    def make_complex_inputs(self):
        """Generate input files for complex prediction using pre-computed features."""
        check_exist(self._get_inputs_dir())
        
        job_index = 0
        for bait_index, (bait_id, bait_seq, bait_type) in enumerate(self.bait.iter_rows()):
            # Load bait features if available
            bait_path = os.path.join(self.feature_path, bait_id, f"{bait_id}_data.json")
            if not os.path.exists(bait_path):
                print(f"Feature folder for {bait_id} not found, skipping...")
                continue
                
            with open(bait_path, "r") as f:
                bait_feature = json.loads(f.read())["sequences"][0]
                
            for prey_index, (prey_id, prey_seq, prey_type) in enumerate(self.prey.iter_rows()):
                # Load prey features if available
                prey_path = os.path.join(self.feature_path, prey_id, f"{prey_id}_data.json")
                if not os.path.exists(prey_path):
                    print(f"Feature folder for {prey_id} not found, skipping...")
                    continue
                    
                with open(prey_path, "r") as f:
                    prey_feature = json.loads(f.read())["sequences"][0]
                
                # Set chain ID for prey
                if self.prey_type in ["ligand_ccd", "ligand_smiles"]:
                    prey_feature["ligand"]["id"] = "B"
                else:
                    prey_feature[prey_type]["id"] = "B"
                
                name = f"{bait_id}-{prey_id}"
                if os.path.exists(os.path.join(self.destination, name)):
                    print(f"{name} already exists, skipping...")
                    continue
                
                target_dir = self._get_job_dir(job_index // self.protein_per_job)
                os.makedirs(target_dir, exist_ok=True)
                job_index += 1
                
                # Create and save complex input
                input_json = json.loads(
                    build_dimer(name, bait_type, str(bait_seq), prey_type, str(prey_seq))
                )
                input_json["sequences"] = [bait_feature, prey_feature]
                
                with open(os.path.join(target_dir, f"{name}.json"), "w") as f:
                    f.write(json.dumps(input_json))

    def make_both_inputs(self):
        """Generate input files for combined feature + complex prediction."""
        check_exist(self._get_inputs_dir())
        
        job_index = 0
        for bait_index, (bait_id, bait_seq, bait_type) in enumerate(self.bait.iter_rows()):
            for prey_index, (prey_id, prey_seq, prey_type) in enumerate(self.prey.iter_rows()):
                name = f"{bait_id}-{prey_id}"
                if os.path.exists(os.path.join(self.destination, name)):
                    print(f"{name} already exists, skipping...")
                    continue
                
                target_dir = self._get_job_dir(job_index // self.protein_per_job)
                os.makedirs(target_dir, exist_ok=True)
                job_index += 1
                
                input_json = build_dimer(name, bait_type, str(bait_seq), prey_type, str(prey_seq))
                with open(os.path.join(target_dir, f"{name}.json"), "w") as f:
                    f.write(input_json)

    def check_job(self):
        """Check job status and report statistics."""
        if self.job_type == "make_feature":
            combined = pl.concat([self.bait, self.prey]).unique()
            status = np.zeros(len(combined))
            
            for index, (id, _, _) in enumerate(combined.iter_rows()):
                if os.path.exists(os.path.join(self.destination, id)):
                    print(f"{id} done")
                    status[index] = 1
                else:
                    print(f"{id} not done")
                    
            print(f"Total jobs: {len(combined)}, Completed: {int(np.sum(status))}, Failed: {len(combined) - int(np.sum(status))}")
            
            if np.sum(status) < len(combined):
                print("Failed jobs:")
                failed = combined.filter(pl.Series(status) == 0)
                for id, _, _ in failed.iter_rows():
                    print(f"  {id}")
                    
        else:  # make_complex or both
            status = np.zeros((len(self.bait), len(self.prey)))
            
            for b_idx, (bait_id, _, _) in enumerate(self.bait.iter_rows()):
                for p_idx, (prey_id, _, _) in enumerate(self.prey.iter_rows()):
                    name = f"{bait_id}-{prey_id}"
                    if os.path.exists(os.path.join(self.destination, name)):
                        print(f"{name} done")
                        status[b_idx, p_idx] = 1
                    else:
                        print(f"{name} not done")
            
            total = len(self.bait) * len(self.prey)
            completed = int(np.sum(status))
            print(f"Total jobs: {total}, Completed: {completed}, Failed: {total - completed}")
            
            if completed < total:
                print("Failed jobs:")
                for b_idx, p_idx in np.argwhere(status == 0):
                    print(f"  {self.bait[int(b_idx), 'id']}-{self.prey[int(p_idx), 'id']}")

    def check_stat(self):
        """Generate and save statistics for completed jobs."""
        stat_dest = os.path.join(self.destination, "statistics")
        os.makedirs(stat_dest, exist_ok=True)
        
        df = collect_statistics((list(self.bait["id"]), list(self.prey["id"])), self.destination)
        plot_confidence_boxplot(df, os.path.join(stat_dest, "statistics.png"))
        df.write_csv(os.path.join(stat_dest, "statistics.csv"))
        print(f"Statistics saved to {stat_dest}")

    def run(self):
        """Execute the AlphaFold3 workflow."""
        print(f"Running AlphaFold3 {self.job_type} job: {self.job_name}")
        
        # Skip job submission if only checking
        if not any([self.flag_check, self.flag_detailed, self.flag_stat]):
            try:
                if self.job_type == "make_feature":
                    self.make_protein_features_inputs()
                elif self.job_type == "make_complex":
                    if not self.feature_path:
                        self.feature_path = self.destination
                        print("Feature path not provided, using destination path")
                    self.make_complex_inputs()
                elif self.job_type == "both":
                    self.make_both_inputs()
                
                script_path = self.print_script()
                self._sbatch_submit(script_path)
            except Exception as e:
                print(f"Error preparing or submitting job: {e}")

        if self.flag_check:
            self.check_job()
        if self.flag_detailed:
            self.detailed_check()
        if self.flag_stat:
            self.check_stat()


class Alphafold3WrapperMonomer(BaseAlphafold3):
    """Class to run AlphaFold3 on individual protein sequences."""
    
    def __init__(
        self,
        job_name: str,
        sequences: List[str],
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
    ):
        # This class always uses "both" job type
        job_type = "both"
        
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
        )
        
        # Load sequences
        self.sequences = read_file_as_df(sequences)
        
        # Determine job distribution
        self.max_jobs = min(self.max_jobs, len(self.sequences))
        self.protein_per_job = max(1, len(self.sequences) // self.max_jobs)

    def make_protein_monomer_inputs(self):
        """Generate input files for monomer prediction."""
        check_exist(self._get_inputs_dir())
        
        for index, (id, seq, type) in enumerate(self.sequences.iter_rows()):
            output_dir = os.path.join(self.destination, id)
            if os.path.exists(output_dir):
                print(f"{id} already exists, skipping...")
                continue
                
            job_index = index // self.protein_per_job
            target_dir = self._get_job_dir(job_index)
            os.makedirs(target_dir, exist_ok=True)
            
            input_json = build_monomer(id, type, str(seq))
            with open(os.path.join(target_dir, f"{id}.json"), "w") as f:
                f.write(input_json)

    def check_job(self):
        """Check job status and report statistics."""
        status = np.zeros(len(self.sequences))
        
        for index, (id, _, _) in enumerate(self.sequences.iter_rows()):
            output_dir = os.path.join(self.destination, id)
            if os.path.exists(output_dir):
                print(f"{id} done")
                status[index] = 1
            else:
                print(f"{id} not done")
                
        completed = int(np.sum(status))
        print(f"Total jobs: {len(self.sequences)}, Completed: {completed}, Failed: {len(self.sequences) - completed}")
        
        if completed < len(self.sequences):
            print("Failed jobs:")
            failed = self.sequences.filter(pl.Series(status) == 0)
            for id, _, _ in failed.iter_rows():
                print(f"  {id}")

    def check_stat(self):
        """Generate and save statistics for completed jobs."""
        stat_dest = os.path.join(self.destination, "statistics")
        os.makedirs(stat_dest, exist_ok=True)
        
        df = collect_statistics(list(self.sequences["id"]), self.destination)
        plot_confidence_boxplot(df, os.path.join(stat_dest, "statistics.png"))
        df.write_csv(os.path.join(stat_dest, "statistics.csv"))
        print(f"Statistics saved to {stat_dest}")

    def run(self):
        """Execute the AlphaFold3 workflow."""
        print(f"Running AlphaFold3 monomer job: {self.job_name}")
        
        # Skip job submission if only checking
        if not any([self.flag_check, self.flag_detailed, self.flag_stat]):
            try:
                self.make_protein_monomer_inputs()
                script_path = self.print_script()
                self._sbatch_submit(script_path)
            except Exception as e:
                print(f"Error preparing or submitting job: {e}")

        if self.flag_check:
            print(f"Checking {len(self.sequences)} jobs")
            self.check_job()
        if self.flag_detailed:
            self.detailed_check()
        if self.flag_stat:
            self.check_stat()


class Alphafold3Multimer(BaseAlphafold3):
    """
    Class to run AlphaFold3 multimer (supports 1-26 molecules in a complex).
    Reminder: You must run with make features before running make complex.
    """
    
    def __init__(
        self,
        job_name: str,
        job_type: Literal["make_feature", "make_complex", "both"],
        input_types: List[Literal["protein", "ligand_ccd", "ligand_smiles", "dna", "rna"]],
        input_paths: List[Union[str, List[str]]],
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
        )
        
        # Validate input parameters
        if len(input_types) != len(input_paths):
            raise ValueError("input_types and input_paths must have the same length")
            
        # Load sequences
        self.sequence_lists = [
            read_file_as_df(path, input_type)
            for input_type, path in zip(input_types, input_paths)
        ]

        if exact:
            self.sequence_lists = filter_dataframes(self.sequence_lists)
            
        # Store configuration
        self.feature_path = feature_path
        self.exact = exact
        self.type_list = input_types
        
        # Calculate total job count
        if self.job_type == "make_feature":  
            self.total_length = sum(len(seq_list) for seq_list in self.sequence_lists)
        elif self.exact:
            self.total_length = len(self.sequence_lists[0])
        else:
            self.total_length = np.prod([len(seq_list) for seq_list in self.sequence_lists])
            
        # Determine job distribution
        self.max_jobs = min(self.max_jobs, self.total_length) if self.total_length > 0 else self.max_jobs
        self.protein_per_job = max(1, self.total_length // self.max_jobs)

    def _get_combinations(self):
        """Get appropriate combinations of sequences based on exact flag."""
        if self.exact:
            return zip(*map(lambda x: x.iter_rows(), self.sequence_lists))
        else:
            return product(*map(lambda x: x.iter_rows(), self.sequence_lists))
            
    def _build_complex_name(self, combination):
        """Build complex name from sequence IDs."""
        return "-".join(seq_id for seq_id, _, _ in combination)

    def make_protein_features_inputs(self):
        """Generate input files for feature computation."""
        check_exist(self._get_inputs_dir())
        
        # Combine and deduplicate sequences
        combined = pl.concat(self.sequence_lists).unique().drop_nulls()
        self.combined = combined
        
        for index, (id, seq, type) in tqdm(enumerate(combined.iter_rows())):
            if os.path.exists(os.path.join(self.destination, id)):
                print(f"{id} already exists, skipping...")
                continue
                
            job_index = index // self.protein_per_job
            target_dir = self._get_job_dir(job_index)
            
            input_json = build_monomer(id, type, str(seq))
            os.makedirs(target_dir, exist_ok=True)
            with open(os.path.join(target_dir, f"{id}.json"), "w") as f:
                f.write(input_json)

    def make_complex_inputs(self):
        """Generate input files for complex prediction using pre-computed features."""
        check_exist(self._get_inputs_dir())
        
        job_index = 0
        for index, combination in tqdm(enumerate(self._get_combinations())):
            # Generate complex name
            try:
                name = self._build_complex_name(combination)
            except Exception as e:
                print(f"Error building complex name: {e}")
                continue
            
            # Skip if output already exists
            if os.path.exists(os.path.join(self.destination, name)) and any(
                file.endswith(".cif") for file in os.listdir(os.path.join(self.destination, name))
            ):
                print(f"{name} already exists, skipping...")
                continue
                
            # Collect features for each component
            chain_id = "A"
            features = []
            feature_missing = False
            
            for seq_id, seq, seq_type in combination:
                # Check if features exist
                if not os.path.exists(os.path.join(self.feature_path, seq_id)):
                    print(f"Features of {seq_id} does not exist, skipping...")
                    feature_missing = True
                    break
                    
                # Load features
                feature_path = os.path.join(self.feature_path, seq_id, f"{seq_id}_data.json")
                with open(feature_path, "r") as f:
                    feature = json.loads(f.read())["sequences"][0]
                
                # Set chain ID
                if chain_id >= "Z":
                    raise ValueError("Too many sequences, should be less than 26")
                    
                if seq_type in ["ligand_ccd", "ligand_smiles"]:
                    feature["ligand"]["id"] = chain_id
                else:
                    feature[seq_type]["id"] = chain_id
                    
                chain_id = chr(ord(chain_id) + 1)
                features.append(feature)
                
            if feature_missing:
                continue
                
            # Create job directory
            target_dir = self._get_job_dir(job_index // self.protein_per_job)
            os.makedirs(target_dir, exist_ok=True)
            job_index += 1
            
            # Create and save complex input
            input_json = json.loads(
                build_multimer(name, self.type_list, [""] * len(self.type_list), is_feature=True)
            )
            input_json["sequences"] = features
            
            with open(os.path.join(target_dir, f"{name}.json"), "w") as f:
                f.write(json.dumps(input_json))

    def make_both_inputs(self):
        """Generate input files for combined feature + complex prediction."""
        check_exist(self._get_inputs_dir())
        
        job_index = 0
        for index, combination in tqdm(enumerate(self._get_combinations())):
            # Generate complex name and collect sequences
            name = self._build_complex_name(combination)
            seqs = [str(seq) for _, seq, _ in combination]
            
            # Skip if output already exists
            if os.path.exists(os.path.join(self.destination, name)) and any(
                file.endswith(".cif") for file in os.listdir(os.path.join(self.destination, name))
            ):
                print(f"{name} already exists, skipping...")
                continue
                
            # Create job directory
            target_dir = self._get_job_dir(job_index // self.protein_per_job)
            os.makedirs(target_dir, exist_ok=True)
            job_index += 1
            
            # Create and save input file
            input_json = build_multimer(name, self.type_list, seqs)
            with open(os.path.join(target_dir, f"{name}.json"), "w") as f:
                f.write(input_json)

    def check_job(self):
        """Check job status and report statistics."""
        if self.job_type == "make_feature":
            # For feature stage, check each unique sequence
            combined = pl.concat(self.sequence_lists).unique()
            status = np.zeros(len(combined))
            
            for index, (id, _, _) in enumerate(combined.iter_rows()):
                if os.path.exists(os.path.join(self.destination, id)):
                    print(f"{id} done")
                    status[index] = 1
                else:
                    print(f"{id} not done")
                    
            completed = int(np.sum(status))
            print(f"Total jobs: {len(combined)}, Completed: {completed}, Failed: {len(combined) - completed}")
            
            if completed < len(combined):
                print("Failed jobs:")
                failed = combined.filter(pl.Series(status) == 0)
                for id, _, _ in failed.iter_rows():
                    print(f"  {id}")
                    
        else:  # make_complex or both
            # For complex stage, check all combinations
            status = np.zeros(self.total_length)
            name_list = []
            
            for index, combination in enumerate(self._get_combinations()):
                name = self._build_complex_name(combination)
                name_list.append(name)
                
                output_dir = os.path.join(self.destination, name)
                if os.path.exists(output_dir) and any(
                    file.endswith(".cif") for file in os.listdir(output_dir)
                ):
                    print(f"{name} done")
                    status[index] = 1
                else:
                    print(f"{name} not done")
            
            completed = int(np.sum(status))
            print(f"Total jobs: {self.total_length}, Completed: {completed}, Failed: {self.total_length - completed}")
            
            if completed < self.total_length:
                print("Failed jobs:")
                for i in np.argwhere(status == 0).flatten():
                    print(f"  {name_list[int(i)]}")

    def check_stat(self):
        """Generate and save statistics for completed jobs."""
        stat_dest = os.path.join(self.destination, "statistics")
        os.makedirs(stat_dest, exist_ok=True)
        
        # Collect ID lists for statistics
        id_lists = [list(seq_list["id"]) for seq_list in self.sequence_lists]
        
        if self.exact:
            df = collect_statistics_exact(id_lists, self.destination)
        else:
            df = collect_statistics(id_lists, self.destination)
            
        plot_confidence_boxplot(df, os.path.join(stat_dest, "statistics.png"))
        df.write_csv(os.path.join(stat_dest, "statistics.csv"))
        print(f"Statistics saved to {stat_dest}")

    def run(self):
        """Execute the AlphaFold3 workflow."""
        print(f"Running AlphaFold3 multimer {self.job_type} job: {self.job_name}")
        
        # Skip job submission if only checking
        if not any([self.flag_check, self.flag_detailed, self.flag_stat]):
            try:
                if self.job_type == "make_feature":
                    self.make_protein_features_inputs()
                elif self.job_type == "make_complex":
                    if not self.feature_path:
                        self.feature_path = self.destination
                        print("Feature path not provided, using destination path")
                    self.make_complex_inputs()
                elif self.job_type == "both":
                    self.make_both_inputs()
                
                script_path = self.print_script()
                self._sbatch_submit(script_path)
            except Exception as e:
                print(f"Error preparing or submitting job: {e}")

        if self.flag_check:
            self.check_job()
        if self.flag_detailed:
            self.detailed_check()
        if self.flag_stat:
            self.check_stat()