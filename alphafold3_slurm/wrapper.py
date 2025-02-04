from .input_utils import (build_protein_dimer, 
                         build_monomer, 
                         read_file_as_df, 
                         build_dimer, 
                         check_exist)
from .stat_utils import collect_statistics, plot_confidence_boxplot
from .config import Config
from typing import Literal, List
import json
import polars as pl
import numpy as np
import os
import shutil

config = Config()

class BaseAlphafold3:
    def __init__(self, job_name: str, job_type: str, destination: str, num_sample: int, time_each_protein: int, max_jobs: int, memory: int, num_cpu: int, gpu_type: Literal["a100", "v100"], flag_check: bool, flag_detailed: bool, flag_stat: bool, email: str):
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

    def _compute_time(self, num_items: int):
        total_minutes = self.time_each_protein * np.ceil(num_items / self.max_jobs)
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        seconds = int((total_minutes - int(total_minutes)) * 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    
    def print_script(self):
        """
        Function to print the script to run AlphaFold3, this script is for SLURM (Currently only for Ibex)
        """
        gpu_parameter = ""
        model_dir = config["parameter"]
        db_dir = config["db"]
        env_dir = config["env"]
        if self.job_type == "make_feature":
            python_script = f"run_alphafold --json_path=$json --model_dir={model_dir} --db_dir={db_dir} --output_dir={self.destination} --norun_inference"
        elif self.job_type == "make_complex":
            gpu_parameter = "\n".join(["#SBATCH --gres=gpu:1",
                        f"#SBATCH --constraint={self.gpu_type}"])
            python_script = f"run_alphafold --json_path=$json --model_dir={model_dir} --db_dir={db_dir} --num_diffusion_samples {self.num_sample} --output_dir={self.destination} --norun_data_pipeline"
        elif self.job_type == "both":
            gpu_parameter = "\n".join(["#SBATCH --gres=gpu:1",
                        f"#SBATCH --constraint={self.gpu_type}"])
            python_script = f"run_alphafold --json_path=$json --model_dir={model_dir} --db_dir={db_dir} --num_diffusion_samples {self.num_sample} --output_dir={self.destination}"
        else:
            raise ValueError("job_type must be either make_feature, make_complex, or both")
        
        job_num = len(os.listdir(os.path.join(self.destination, f"inputs_{self.job_type}")))
            
        script = "\n".join([
                "#!/bin/bash",
                "#SBATCH -N 1",
                f"#SBATCH --array 0-{job_num - 1}",
                f"#SBATCH --job-name={self.job_name}",
                f"#SBATCH --output={os.path.join(self.destination, 'ibex_out', '%x-%j.out')}",
                f"#SBATCH --time={self._compute_time(len(os.listdir(os.path.join(self.destination, f'inputs_{self.job_type}', 'job-0'))))}",
                f"#SBATCH --mem={self.memory}G",
                f"#SBATCH --cpus-per-task={self.num_cpu}", 
                f"#SBATCH --mail-type=ALL" if self.email is not None else "",
                f"#SBATCH --mail-user={self.email}" if self.email is not None else "",
                gpu_parameter, 
                "source ~/.bashrc",
                f"conda activate {env_dir}",
                "export CUDA_VISIBLE_DEVICES=0,1,2,3",
                "export TF_FORCE_UNIFIED_MEMORY=1",
                "export LA_FLAGS=\"--xla_gpu_enable_triton_gemm=false\"",
                "export XLA_PYTHON_CLIENT_PREALLOCATE=true",
                "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95",
                f"for json in {os.path.join(self.destination, f'inputs_{self.job_type}', 'job-$SLURM_ARRAY_TASK_ID', '*.json')}; do",
                "echo $json",
                f"echo {self.job_type}",
                "echo SOJ_indicator",
                f"time {python_script}",
                "echo $?",
                "echo EOJ_indicator",
                "done"
                ])
        
        os.makedirs(os.path.join(self.destination, "script"), exist_ok=True)
        script_path = os.path.join(self.destination, "script", f"run-{self.job_type}.slurm")
        open(script_path, "w").write(script)
        return script_path

    def _sbatch_submit(self, script_path: str):
        output = os.popen(f"sbatch {script_path}").read()
        print(output)
        print(f"Job submitted")

    def detailed_check(self):
        for file_name in os.listdir(os.path.join(self.destination, "ibex_out")):
            with open(os.path.join(self.destination, "ibex_out", file_name), "r") as f:
                lines = f.readlines()
                for index, line in enumerate(lines):
                    if "SOJ_indicator" in line:
                        current_file = lines[index - 2]
                        current_type = lines[index - 1]
                        if current_type != self.job_type:
                            break
                        continue
                    if "EOJ_indicator" in line:
                        if "0" in lines[index - 1]:
                            print(f"{current_file} done")
                        else:
                            print(f"{current_file} failed, last lines:")
                            for i in range(5, 0 , -1):
                                print(lines[index - 1 - i])
                            current_file_tailless = os.path.splitext(current_file)[0]
                            if os.path.exists(os.path.join(self.destination, current_file_tailless)):
                                shutil.rmtree(os.path.join(self.destination, current_file_tailless), ignore_errors=True)

class Alphafold3PullDown(BaseAlphafold3):
    """
    Class to run AlphaFold3 on Ibex
    Reminder: You must run with make features before running make complex.
    """
    def __init__(self, job_name: str, job_type: Literal["make_feature", "make_complex", "both"], bait_type: Literal["protein", "ligand_ccd", "ligand_smiles", "dna", "rna"], bait_path: str|List[str], prey_type: Literal["protein", "ligand_ccd", "ligand_smiles", "dna", "rna"], prey_path: str|List[str], \
                 destination: str, feature_path: str=None, num_sample:int=5, time_each_protein:int=120, max_jobs: int=1500, memory: int=64, num_cpu:int=8, gpu_type: Literal["a100", "v100"]="a100", flag_check: bool=False, flag_detailed=False, flag_stat: bool=False, email: str=None):
        """
        Class to make inputs for AlphaFold3
        :param job_name: name of the job
        :param job_type: type of the job, either "make_feature", "make_complex", or "both"
        :param bait_path: path to the bait sequence in fasta format
        :param prey_path: path to the prey sequences in fasta format
        :param destination: path to the destination folder
        :param feature_path: path to the feature file, it should be the same as the destination folder in make_feature stage
        :param time_each_protein: time needed to run each protein
        :param memory: memory needed to run the job
        :param max_jobs: maximum number of jobs to run
        :param num_cpu: number of CPUs to use
        :param gpu_type: type of GPU to use, either "a100" or "v100"
        :param flag_check: flag to check only instead of running the job
        :param flag_stat: flag to show the statistics of the job
        """
        super().__init__(job_name, job_type, destination, num_sample, time_each_protein, max_jobs, memory, num_cpu, gpu_type, flag_check, flag_detailed, flag_stat, email)
        self.job_name = job_name
        self.job_type = job_type
        self.bait = read_file_as_df(bait_path, bait_type)
        self.prey = read_file_as_df(prey_path, prey_type)
        self.destination = destination
        self.feature_path = feature_path
        self.max_jobs = max_jobs
        self.memory = memory
        self.num_cpu = num_cpu
        self.time_each_protein = time_each_protein
        self.prey_type = prey_type
        self.num_sample = num_sample
        self.bait_type = bait_type
        if len(self.prey) < self.max_jobs:
            self.max_jobs = len(self.prey)
        self.protein_per_job = len(self.prey) // self.max_jobs
        self.gpu_type = gpu_type
        self.flag_check = flag_check
        self.flag_detailed = flag_detailed
        self.flag_stat = flag_stat
        self.email = email

    def make_protein_features_inputs(self):
        """
        Function to make inputs for AlphaFold3, this is for make feature stage
        :param sequences: sequences in fasta format
        """
        check_exist(os.path.join(self.destination, f"inputs_{self.job_type}"))
        combined = pl.concat([self.bait, self.prey])
        self.combined = combined.unique()
        for index, (id, seq, type) in enumerate(self.combined.iter_rows()):
            if os.path.exists(os.path.join(self.destination, id)):
                print(f"{id} already exists, skipping...")
                continue
            target_dir = os.path.join(self.destination, f"inputs_{self.job_type}", f"job-{index // self.protein_per_job}")
            os.makedirs(target_dir, exist_ok=True)
            input_json = build_monomer(id, type, str(seq))
            with open(os.path.join(target_dir, f"{id}.json"), "w") as f:
                f.write(input_json)

    def make_complex_inputs(self):
        """
        Function to make inputs for AlphaFold3. this is for make complex stage
        :param bait: bait sequence in fasta format, should be a single sequence
        :param prey: prey sequences in fasta format, should be multiple sequences
        """
        check_exist(os.path.join(self.destination, f"inputs_{self.job_type}"))
        for bait_index, (bait_id, bait_seq, bait_type) in enumerate(self.bait.iter_rows()):
            bait_path = os.path.join(self.feature_path, bait_id, f"{bait_id}_data.json")
            if not os.path.exists(bait_path):
                print(f"Feature folder for {bait_id} not found, skipping...")
                continue
            with open(bait_path, "r") as f:
                bait_feature = json.loads(f.read())["sequences"][0]
            for prey_index, (prey_id, prey_seq, prey_type) in enumerate(self.prey.iter_rows()):
                prey_path = os.path.join(self.feature_path, prey_id, f"{prey_id}_data.json")
                if not os.path.exists(prey_path):
                    print(f"Feature folder for {prey_id} not found, skipping...")
                    continue
                with open(prey_path, "r") as f:
                    prey_feature = json.loads(f.read())["sequences"][0]
                if self.prey_type == "ligand_ccd" or self.prey_type == "ligand_smiles":
                    bait_feature["ligand"]["id"] = "B"
                else:
                    prey_feature[prey_type]["id"] = "B"
                name = f"{bait_id}-{prey_id}"
                if os.path.exists(os.path.join(self.destination, name)):
                    print(f"{name} already exists, skipping...")
                    continue
                index = bait_index * len(self.prey) + prey_index
                target_dir = os.path.join(self.destination, f"inputs_{self.job_type}", f"job-{index // self.protein_per_job}")
                os.makedirs(target_dir)
                input_json = json.loads(build_dimer(name, bait_type, str(bait_seq), prey_type, str(prey_seq)))
                input_json["sequences"] = [bait_feature, prey_feature]
                input_json = json.dumps(input_json)
                with open(os.path.join(target_dir, f'{name}.json'), "w") as f:
                    f.write(input_json)
    
    def make_both_inputs(self):
        """
        Function to make inputs for AlphaFold3, this is for job type "both"
        :param bait: bait sequence in fasta format, should be a single sequence
        :param prey: prey sequences in fasta format, should be multiple sequences
        """
        check_exist(os.path.join(self.destination,f"inputs_{self.job_type}"))
        for bait_index, (bait_id, bait_seq, bait_type) in enumerate(self.bait.iter_rows()):
            for prey_index, (prey_id, prey_seq, prey_type) in enumerate(self.prey.iter_rows()):
                name = f"{bait_id}-{prey_id}"
                if os.path.exists(os.path.join(self.destination, name)):
                    print(f"{name} already exists, skipping...")
                    continue
                index = bait_index * len(self.prey) + prey_index
                target_dir = os.path.join(self.destination, f"inputs_{self.job_type}", f"job-{index // self.protein_per_job}")
                os.makedirs(target_dir, exist_ok=True)
                input_json = build_dimer(name, bait_type, str(bait_seq), prey_type, str(prey_seq))
                with open(os.path.join(target_dir, f"{name}.json"), "w") as f:
                    f.write(input_json)

    def check_job(self):
        """
        Function to check the job status, in status dict, 1 means done, 0 means not done
        """
        output = ""
        if self.job_type == "make_feature":
            status = np.zeros(len(self.bait) + len(self.prey))
            combined = pl.concat([self.bait, self.prey])
            for index, (id, seq, type) in enumerate(combined.iter_rows()):
                target_dir = os.path.join(self.destination, id)
                if os.path.exists(target_dir):
                    print(f"{id} done")
                    status[index] = 1
                else:
                    print(f"{id} not done")
            output = f"Total number of jobs: {status.shape[1]}\n Number of jobs done: {int(np.sum(status))}, Number of jobs not done: {len(combined) - np.sum(status)}"
            print(output)
            failed_jobs = combined.filter(pl.Series(status) == 0)
            print("Failed jobs:")
            for index, (id, seq, type) in enumerate(failed_jobs.iter_rows()):
                print(f"{id} failed")
                           
        elif self.job_type == "make_complex" or self.job_type == "both":
            status = np.zeros((len(self.bait), len(self.prey)))
            for bait_index, (bait_id, bait_seq, bait_type) in enumerate(self.bait.iter_rows()):
                for prey_index, (prey_id, prey_seq, prey_type) in enumerate(self.prey.iter_rows()):
                    name = f"{bait_id}-{prey_id}"
                    target_dir = os.path.join(self.destination, name)
                    if os.path.exists(target_dir):
                        print(f"{name} done")
                        status[bait_index, prey_index] = 1
                    else:
                        print(f"{name} not done")
            output = f"Total number of jobs: {status.shape[1]}\n Number of jobs done: {int(np.sum(status))}, Number of jobs not done: {int(len(self.bait) * len(self.prey) - np.sum(status))}"
            print(output)
            failed_jobs = np.argwhere(status == 0)
            print("Failed jobs:")
            for bait_index, prey_index in failed_jobs:
                print(f"{self.bait[bait_index][0]}-{self.prey[prey_index][0]} failed")

    def check_stat(self):
        """
        Function to check the statistics of the job
        """
        stat_dest = os.path.join(self.destination, "statistics")
        os.makedirs(stat_dest, exist_ok=True)
        df = collect_statistics((list(self.bait["id"]), list(self.prey["id"])), self.destination)
        plot_confidence_boxplot(df, os.path.join(stat_dest, "statistics.png"))
        df.write_csv(os.path.join(stat_dest, "statistics.csv"))
                    
    def run(self):
        """
        Function to run the AlphaFold3 job
        """
        print("Running AlphaFold3 job")
        if self.flag_check or self.flag_detailed or self.flag_stat:
            print("Checking only, no job submitted")
        else:
            if self.job_type == "make_feature":
                self.make_protein_features_inputs()
                self._sbatch_submit(self.print_script())
            elif self.job_type == "make_complex":
                if self.feature_path is None:
                    print("Feature path is not provided, will use destination path instead")
                    self.feature_path = self.destination
                self.make_complex_inputs()
                self._sbatch_submit(self.print_script())
            elif self.job_type == "both":
                self.make_both_inputs()
                self._sbatch_submit(self.print_script())
        
        if self.flag_check:
            print(f"Total number of jobs: {len(self.prey)}")
            self.check_job()
        if self.flag_detailed:
            print("Checking detailed status")
            self.detailed_check()
        if self.flag_stat:
            print("Checking statistics")
            self.check_stat()

class Alphafold3WrapperMonomer(BaseAlphafold3):
    def __init__(self, job_name: str, sequences: List[str], destination: str, num_sample: int = 5, time_each_protein: int = 120, max_jobs: int = 1500, memory: int = 64, num_cpu: int = 8, gpu_type: Literal["a100", "v100"] = "a100", flag_check: bool = False, flag_detailed: bool = False, flag_stat: bool = False, email: str = None):
        """
        Class to make inputs for AlphaFold3 from user-provided sequences
        :param sequences: list of sequences in fasta format
        :param destination: path to the destination folder
        :param time_each_protein: time needed to run each protein
        :param memory: memory needed to run the job
        :param max_jobs: maximum number of jobs to run
        :param num_cpu: number of CPUs to use
        :param gpu_type: type of GPU to use, either "a100" or "v100"
        :param flag_check: flag to check only instead of running the job
        :param flag_stat: flag to show the statistics of the job
        """
        self.job_name = job_name
        self.job_type = "both"
        super().__init__(self.job_type, destination, num_sample, time_each_protein, max_jobs, memory, num_cpu, gpu_type, flag_check, flag_detailed, flag_stat, email)
        self.sequences = read_file_as_df(sequences)
        self.destination = destination
        self.max_jobs = max_jobs
        self.memory = memory
        self.num_cpu = num_cpu
        self.time_each_protein = time_each_protein
        self.num_sample = num_sample
        if len(self.sequences) < self.max_jobs:
            self.max_jobs = len(self.sequences)
        self.protein_per_job = len(self.sequences) // self.max_jobs
        self.gpu_type = gpu_type
        self.flag_check = flag_check
        self.flag_detailed = flag_detailed
        self.flag_stat = flag_stat
        self.email = email

    def make_protein_monomer_inputs(self):
        """
        Function to make inputs for AlphaFold3, this is for make feature stage
        """
        check_exist(os.path.join(self.destination, f"inputs_{self.job_type}"))
        combined = pl.concat([self.bait, self.prey])
        self.combined = combined.unique()
        for index, (id, seq, type) in enumerate(self.combined.iter_rows()):
            if os.path.exists(os.path.join(self.destination, id)):
                print(f"{id} already exists, skipping...")
                continue
            target_dir = os.path.join(self.destination, f"inputs_{self.job_type}", f"job-{index // self.protein_per_job}")
            os.makedirs(target_dir, exist_ok=True)
            input_json = build_monomer(id, type, str(seq))
            with open(os.path.join(target_dir, f"{id}.json"), "w") as f:
                f.write(input_json)

    def check_job(self):
        """
        Function to check the job status, in status dict, 1 means done, 0 means not done
        """
        status = np.zeros(len(self.sequences))
        for index, seq in enumerate(self.sequences):
            id = f"seq_{index}"
            target_dir = os.path.join(self.destination, id)
            if os.path.exists(target_dir):
                print(f"{id} done")
                status[index] = 1
            else:
                print(f"{id} not done")
        output = f"Total number of jobs: {len(self.sequences)}\n Number of jobs done: {int(np.sum(status))}, Number of jobs not done: {len(self.sequences) - np.sum(status)}"
        print(output)
        failed_jobs = [seq for i, seq in enumerate(self.sequences) if status[i] == 0]
        print("Failed jobs:")
        for seq in failed_jobs:
            print(f"{seq} failed")

    def check_stat(self):
        """
        Function to check the statistics of the job
        """
        stat_dest = os.path.join(self.destination, "statistics")
        os.makedirs(stat_dest, exist_ok=True)
        df = collect_statistics(list(self.sequences["id"]), self.destination)
        plot_confidence_boxplot(df, os.path.join(stat_dest, "statistics.png"))
        df.write_csv(os.path.join(stat_dest, "statistics.csv"))
                    
    def run(self):
        """
        Function to run the AlphaFold3 job
        """
        print("Running AlphaFold3 job")
        if self.flag_check or self.flag_detailed or self.flag_stat:
            print("Checking only, no job submitted")
        else:
            self.make_protein_monomer_inputs()
            self._sbatch_submit(self.print_script())
        
        if self.flag_check:
            print(f"Total number of jobs: {len(self.sequences)}")
            self.check_job()
        if self.flag_detailed:
            print("Checking detailed status")
            self.detailed_check()
        if self.flag_stat:
            print("Checking statistics")
            self.check_stat()