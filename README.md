# Alphafold3 Slurm Pipeline
This repository contains a pipeline for running large scale screening AlphaFold3 on a Slurm cluster. Some concepts are based on the [AlphaFold2 pipeline](https://github.com/strubelab/alphafold). It currently provides the following features:
- Make features: Generate features for two list of protein sequences, extracting MSAs and templates.
- Make complexes: Generate dimer complexes for two list of protein sequences, based on features.
- Make monomer: Generate monomer models for a list of protein sequences listed in a fasta file.

To install it, please follow the instructions below (**This installation guide is for Ibex system of KAUST only. For non-Ibex users, please refer to the [Non-ibex User](#non-ibex) part.**).
## Installation
1. Clone this repository.
2. `cd alphafold3_slurm_pipeline`
3. Run `./af3_env_setup.sh` to install the required dependencies. **This process would submit a sbatch job, and could take fair amount of time.**
4. If the installation is failed, you can refer to "AF_install.out" for any error messages. Please contact the author if you have any questions.

## Non-ibex User
<a name="non-ibex"></a>
For non-Ibex users, you need to make several modifications based on the slurm system you use. The following steps are required:
1. Modify alphafold3_slurm/config.py to set the correct paths for your system, this step is done in the setup script.
2. Modify `module load cuda/12.2 gcc/12.2.0` part of the "af3_install.slurm" file to load the correct modules for your system. Usually modern systems should have CUDA and GCC installed.
3. Modify the `--constraint` and `--gres` arguments according to the design of your system. "af3_install.slurm" and "alphfold3_slurm/wrapper.py" files are the file that you need to modify.
4. You should be able to run the installation script as above after these modifications.

## Weight
It is worth noting that the parameter of AF3 should not be distributed or shared without permission. Therefore, if you are looking for the parameter required by AF3. Please refer to [Obtaining Model Parameters](https://github.com/google-deepmind/alphafold3/tree/main?tab=readme-ov-file) of AF3 github page.

## Usage
After installation, you can use the pipeline to generate features, complexes, and monomers. Activate the environment before running any command (by default, `cd` to this repository and run `conda activate ./env`). To check submitted jobs, run `squeue -u $USER`.

> If a generated `inputs_*` directory already exists, commands now fail fast. Re-run with `--overwrite` to replace the generated inputs for that stage.

### Input submit
The most common usage of this pipeline is to submit one JSON input or a directory of JSON inputs. You can refer to [this document](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md) for details on the AlphaFold3 input format.

To submit every `*.json` file in a directory:
`submit_input --input <input_folder> --output <output_folder> --time <minutes> --mem <gb>`
To submit a single JSON file, pass the file path to `--input`. Outputs are written to the directory specified by `--output`.

> `submit_input` submits jobs immediately; the wrapper commands below generate staged `inputs_*` directories and then submit a Slurm array script.

### Pulldown
These are dedicated for AlphaPulldown-style bait/prey dimers. To check which jobs are done, add `--check_only`. To print detailed failure information from Slurm logs, add `--check_only_exact`. `--check_stat` writes pLDDT, ipTM, and pTM statistics for completed complexes.

`af3pulldown --job_name JOB_NAME --bait_type TYPE --bait_input FILE1 [FILE2...] --prey_type TYPE --prey_input FILE1 [FILE2...] --destination OUTPUT_DIR [--feature_path FEATURES] [--make_feature|--make_complex|--make_both] [--overwrite]`

> When running `--make_complex` without `--feature_path`, the command looks for features in `--destination`.

> `--overwrite` replaces the generated `inputs_<stage>` directory only. Existing completed model outputs are still skipped automatically.

Required Arguments
- `--job_name`: Name for the IBEX job (default: `AF3_pulldown`)
- `--destination`: Output directory path for features/structures and job files
- Exactly one of these flags must be specified:
  - `--make_feature`: Generate only AlphaFold features
  - `--make_complex`: Generate only complex predictions (requires existing features)
  - `--make_both`: Generate features then predict complexes (slower than running separately)

> `--feature_path` is only valid with `--make_complex`.

> Input files may be FASTA, CSV, or TSV.

Input Arguments
- `--bait_type`: Type of bait molecule (`protein`, `ligand_ccd`, `ligand_smiles`, `dna`, `rna`)
- `--bait_input`: One or more input files for bait sequences
- `--prey_type`: Type of prey molecule (`protein`, `ligand_ccd`, `ligand_smiles`, `dna`, `rna`)
- `--prey_input`: One or more input files for prey sequences
- `--feature_path`: Directory containing pre-generated features

> Completed feature outputs are detected by `<name>/<name>_data.json`; completed structure outputs are detected by generated `.cif` files.

Optional Arguments
- `--time`: Minutes per job (default: `300`)
- `--mem`: GB memory per job (default: `64`)
- `--mail`: Email for job notifications
- `--gpu_type`: GPU architecture to use (`a100` or `v100`, default: `a100`)
- `--max_jobs`: Maximum concurrent jobs (default: `1990`)
- `--overwrite`: Replace an existing generated `inputs_make_feature`, `inputs_make_complex`, or `inputs_both` directory before writing new inputs
- `--check_only`: Check completion status only
- `--check_only_exact`: Check and report detailed errors
- `--check_stat`: Print pLDDT, ipTM, and pTM statistics

> `--check_only` and `--check_only_exact` never submit jobs.

### Oligomer
This script manages AlphaFold3 predictions for multi-molecule complexes. For example, if the inputs are `molecule_list_A.fasta molecule_list_B.fasta molecule_list_C.fasta`, the results are `A1-B1-C1`, `A1-B1-C2`, ..., `Al-Bm-Cn`.

`af3oligomer --job_name JOB_NAME --input_type TYPE1 [TYPE2...] --input FILE1 [FILE2...] --destination OUTPUT_DIR [--feature_path FEATURES] [--make_feature|--make_complex|--make_both] [--overwrite]`

> `--exact` pairs the nth row from each input instead of taking the Cartesian product. All input lists must have the same length when `--exact` is set.

> Input files may be FASTA, CSV, or TSV.

Required Arguments
- `--job_name`: Name for the IBEX job (default: `AF3_oligomer`)
- `--destination`: Output directory path for features/structures and job files
- Exactly one of these flags must be specified:
  - `--make_feature`: Generate only AlphaFold features
  - `--make_complex`: Generate only complex predictions (requires existing features)
  - `--make_both`: Generate features then predict complexes (slower than running separately)

Input Arguments
- `--input_type`: Types of molecules in order (`protein`, `ligand_ccd`, `ligand_smiles`, `dna`, `rna`)
- `--input`: Input files for sequences in corresponding order
- `--feature_path`: Directory containing pre-generated features

> `--feature_path` is only valid with `--make_complex`.

> In `--make_feature` mode, duplicate sequences across inputs are generated once and reused by later complex stages.

Optional Arguments
- `--time`: Minutes per job (default: `600`)
- `--exact`: Pair nth rows directly instead of taking the Cartesian product
- `--mem`: GB memory per job (default: `64`)
- `--mail`: Email for job notifications
- `--gpu_type`: GPU architecture to use (`a100` or `v100`, default: `a100`)
- `--max_jobs`: Maximum concurrent jobs (default: `1990`)
- `--overwrite`: Replace an existing generated `inputs_make_feature`, `inputs_make_complex`, or `inputs_both` directory before writing new inputs
- `--check_only`: Check completion status only
- `--check_only_exact`: Check and report detailed errors
- `--check_stat`: Print pLDDT, ipTM, and pTM statistics

### Monomer
This script submits jobs to IBEX for protein monomer predictions.

`make_monomers --job_name JOB_NAME --input FILE1 [FILE2...] --destination OUTPUT_DIR [--overwrite]`

> Input files may be FASTA, CSV, or TSV.

Required Arguments
- `--job_name`: Name for the IBEX job
- `--input`: One or more input files (FASTA/CSV/TSV) containing monomer sequences
- `--destination`: Output directory path for predicted structures and job files

Optional Arguments
- `--time`: Minutes per job (default: `300`)
- `--mem`: GB memory per job (default: `64`)
- `--gpu_type`: GPU architecture to use (`a100` or `v100`, default: `a100`)
- `--mail`: Email for job notifications
- `--max_jobs`: Maximum concurrent jobs (default: `1990`)
- `--overwrite`: Replace an existing generated `inputs_both` directory before writing new inputs
- `--check_only`: Check completion status only
- `--check_only_exact`: Check and report detailed errors
- `--check_stat`: Print pLDDT, ipTM, and pTM statistics

You can also display parameter descriptions by calling `<command> --help`. To see more examples, check [examples](examples/example.md).


# Acknowledgement
This tool is partially based on former alphafold wrapper by Javier, the repository is [here](https://github.com/strubelab/alphafold), kudos to him for setting up a standard to follow, and instructions he has provided me. 
Much appreciation for DeepMind for providing such a great tool.