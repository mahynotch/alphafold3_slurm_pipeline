# Alphafold3 Slurm Pipeline
This repository contains a pipeline for running large scale screening AlphaFold3 on a Slurm cluster. Some concepts are based on the [AlphaFold2 pipeline](https://github.com/strubelab/alphafold). It currently provides the following features:
- Make features: Generate features for two list of protein sequences.
- Make complexes: Generate dimer complexes for two list of protein sequences, based on features.
- Make monomer: Generate monomer models for a list of protein sequences.

To install it, please follow the instructions below (**This installation guide is for Ibex system of KAUST only. For non-Ibex users, please refer to the [Non-ibex User](#non-ibex) part.**).
## Installation
1. Clone this repository.
2. `cd alphafold3_slurm_pipeline`
3. Run `./af3_install.sh` to install the required dependencies. **This process would submit a sbatch job, and could take fair amount of time.**
4. If the installation is failed, you can refer to "AF_install.out" for any error messages. Please contact the author if you have any questions.

## Non-ibex User
<a name="non-ibex"></a>
For non-Ibex users, you need to make several modifications based on the slurm system you use. The following steps are required:
1. Modify alphafold3_slurm/config.py to set the correct paths for your system, this step is done in the setup script.
2. Modify `module load cuda/12.2 gcc/12.2.0` part of the "af3_install.slurm" file to load the correct modules for your system. Usually modern systems should have CUDA and GCC installed.
3. Modify the `--constraint` and `--gres` arguments according to the design of your system. "af3_install.slurm" and "alphfold3_slurm/input_utils.py" files are the file that you need to modify.
4. You should be able to run the installation script as above after these modifications.

## Weight
It is worth noting that the parameter of AF3 should not be distributed or shared without permission. Therefore, if you are looking for the parameter required by AF3. Please refer to [Obtaining Model Parameters](https://github.com/google-deepmind/alphafold3/tree/main?tab=readme-ov-file) of AF3 github page.

## Usage
After installation, you can use the pipeline to generate features, complexes and monomers. Please remember that you should run `conda activate <environment directory>` (By default, you can `cd` to this repository and run `conda activate env`) to activate the environment before any script is called. The following is an example of how to use the pipeline:

### Input submit
The most common usage of this pipeline is to submit a input or a list of inputs. You can refer to [this document](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md) for how to create a input file.

To run tests on a directory with a.json, b.json, ..., you only need to run tests on this by the following command:
`submit_input --input <input_folder> --output <output_folder> --time(optional) <time in minutes> --mem(optional) <memory needed>`. To run the script on a single json, all you need to do is replace `<input_folder>` with the path to you target json file. After running this, the outputs will be put in the folder specified in the output parameter.

### Make feature & Make complex
These are dedicated for alphapulldown, taking one bait and one prey to assemble a complex at a time. To check which job is done, you can add `--check_only` at the end. To provide more information on exact reason of failure, you can add `--check_only_exact`. The `--check_stat` only works for complexes, and produce a box plot of plDDT, ipTM, and pTM. The scripts are:

#### make_features:

This script submits jobs to IBEX cluster for generating AlphaFold features (MSAs and templates) for protein sequences and other molecular types.

`make_features --job_name JOB_NAME --bait_type TYPE --bait_input FILE1 [FILE2...] --prey_type TYPE --prey_input FILE1 [FILE2...] --destination OUTPUT_DIR`

Required Arguments

-   --job_name: Name for the IBEX job
-   --destination: Output directory path for AlphaFold features and job files

Input Arguments

-   --bait_type: Type of bait molecule ("protein", "ligand_ccd", "ligand_smiles", "dna", "rna")
-   --bait_input: One or more input files for bait sequences, in csv or fasta
-   --prey_type: Type of prey molecule ("protein", "ligand_ccd", "ligand_smiles", "dna", "rna")
-   --prey_input: One or more input files for prey sequences, in csv or fasta

Optional Arguments

-   --time: Minutes per job (default: 300)
-   --mem: GB memory per job (default: 64)
-   --mail: Email for job notifications
-   --max_jobs: Maximum concurrent jobs (default: 1990)
-   --check_only: Check completion status only
-   --check_only_exact: Check and report detailed errors
-   --check_stat: Print pLDDT, ipTM, and pTM statistics

#### make_complexes

This script submits jobs to IBEX cluster for predicting structures of molecular complexes using AlphaFold.

`make_complexes --job_name JOB_NAME --bait_type TYPE --bait_input FILE1 [FILE2...] --prey_type TYPE --prey_input FILE1 [FILE2...] --destination OUTPUT_DIR --feature_path FEATURES`

Required Arguments

-   --job_name: Name for the IBEX job
-   --destination: Output directory path for redicted structures and job files
-   --feature_path: Directory containing pre-generated features (output from make_feature)

Input Arguments

-   --bait_type: Type of bait molecule ("protein", "ligand_ccd", "ligand_smiles", "dna", "rna")
-   --bait_input: One or more input files for bait sequences
-   --prey_type: Type of prey molecule ("protein", "ligand_ccd", "ligand_smiles", "dna", "rna")
-   --prey_input: One or more input files for prey sequences

Optional Arguments

-   --time: Minutes per job (default: 300)
-   --mem: GB memory per job (default: 64)
-   --mail: Email for job notifications
-   --gpu_type: GPU architecture to use ("a100" or "v100", default: "a100")
-   --max_jobs: Maximum concurrent jobs (default: 1990)
-   --check_only: Check completion status only
-   --check_only_exact: Check and report detailed errors
-   --check_stat: Print pLDDT, ipTM, and pTM statistics


#### make_both
This script combines the two process in one, but is slower than running them seperately. The parameters are similar to that of `make_features`, but you can choose gpu type, so I will skip this.

#### make_monomer
This script submits jobs to IBEX cluster for predicting protein monomer structures using AlphaFold, support fasta file.

`make_monomer --input FILE1 [FILE2...] --destination OUTPUT_DIR`

Required Arguments

-   --input: One or more input files (FASTA/CSV) containing monomer sequences
-   --destination: Output directory path for predicted structures and job files

Optional Arguments

-   --time: Minutes per job (default: 300)
-   --mem: GB memory per job (default: 64)
-   --gpu_type: GPU architecture to use ("a100" or "v100", default: "a100")
-   --mail: Email for job notifications
-   --max_jobs: Maximum concurrent jobs (default: 1990)
-   --check_only: Check completion status only
-   --check_only_exact: Check and report detailed errors
-   --check_stat: Print pLDDT, ipTM, and pTM statistics

# Acknowledgement
This tool is partially based on former alphafold wrapper by Javier, the repository is [here](https://github.com/strubelab/alphafold), kudos to him for setting up a standard to follow, and instructions he has provided me. Much appreciation for DeepMind for providing such great tool.