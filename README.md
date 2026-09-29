# Alphafold3 Slurm Pipeline
This repository contains a pipeline for running large scale screening AlphaFold3 on a Slurm cluster. Some concepts are based on the [AlphaFold2 pipeline](https://github.com/strubelab/alphafold). It currently provides the following features:
- Make features: Generate features for two list of protein sequences, extracting MSAs and templates.
- Make complexes: Generate dimer complexes for two list of protein sequences, based on features.
- Make monomer: Generate monomer models for a list of protein sequences listed in a fasta file.

The pipeline targets **AlphaFold3 v3.0.4** (latest release, July 2026). The installer follows AlphaFold3's own recipe: Python 3.12, HMMER 3.4 with AlphaFold3's `--seq_limit` patch, and `uv sync` against the release's lockfile. Everything goes into one self-contained directory, `af3_env/`. Conda is not used.

## Installation
### Requirements
- A Linux Slurm cluster with NVIDIA GPUs (A100, or V100 with the XLA workarounds the pipeline applies automatically).
- `git`, `curl`, `make`, and a C++ compiler (`g++`). On Ibex, `gcc/12.2.0` is loaded automatically.
- Internet access on the node that runs the installer. `uv` is downloaded automatically if it is not on `PATH`.
- About 15 GB of disk space for `af3_env/`.
- The AlphaFold3 model weights (`af3.bin.zst`). See [Weights](#weights).

### Steps
1. Clone this repository and `cd alphafold3_slurm_pipeline`.
2. Run the installer from the repository root, either as a CPU batch job (recommended; no GPU needed) or directly in your shell:
   ```bash
   sbatch install.sh --params /path/to/weights_dir     # log: AF3_install_<jobid>.out
   # or
   ./install.sh --params /path/to/weights_dir
   ```
   Options:
   | Option | Default | Meaning |
   |---|---|---|
   | `--params DIR` | `./parameter` | Directory containing `af3.bin.zst` |
   | `--db DIR` | `/ibex/reference/KSL/alphafold/3.0.0` | AlphaFold3 genetic databases |
   | `--prefix DIR` | `./af3_env` | Install location |
   | `--af3-version TAG` | `v3.0.4` | AlphaFold3 git tag or commit |
   | `--modules "LIST"` | `gcc/12.2.0` | Environment modules to load before compiling (`""` for none) |
3. Activate the environment when you want to use the commands: `source af3_env/venv/bin/activate`.

Re-running the installer is safe: finished steps are skipped, and the config is rewritten after the old one is backed up. To upgrade AlphaFold3 later, re-run with a new `--af3-version`.

The installer writes `alphafold3_slurm/config.yaml`, which is not tracked by git. To keep a config elsewhere, set `AF3_SLURM_CONFIG=/path/to/config.yaml`. See `config.example.yaml` for the keys.

### Layout
```
af3_env/
├── venv/          # Python 3.12 + AlphaFold3 + this pipeline (editable)
├── alphafold3/    # AlphaFold3 source at the installed tag (run_alphafold.py)
└── hmmer/bin/     # jackhmmer, nhmmer, hmmalign, hmmsearch, hmmbuild
```

### Migrating from the old conda installation
Configs written by the old installer (`db`/`env`/`parameter` only) still work. Running `install.sh` switches the pipeline to the new environment. Once the new install works, you can delete the old `env/` and `alphafold3/` directories.

## Non-Ibex users
<a name="non-ibex"></a>
- Pass `--db` and `--params` to the installer, and `--modules ""` (or your site's compiler module).
- Generated jobs request `--gres=gpu:1` and `--constraint=<gpu_type>`. If your cluster names GPU features differently, adjust `print_script` in `alphafold3_slurm/wrapper.py` and `SCRIPT_TEMPLATE` in `alphafold3_slurm/cli/submit_input.py`.

## Weights
The AlphaFold3 model parameters must not be redistributed. To obtain them, follow [Obtaining Model Parameters](https://github.com/google-deepmind/alphafold3/tree/main?tab=readme-ov-file) on the AlphaFold3 GitHub page. Parameters are compatible with every 3.0.x release.

## Usage
After installation, you can use the pipeline to generate features, complexes, and monomers. Activate the environment before running any command (`source af3_env/venv/bin/activate`). To check submitted jobs, run `squeue -u $USER`.

> If a generated `inputs_*` directory already exists, commands now fail fast. Re-run with `--overwrite` to replace the generated inputs for that stage.

### Input submit
The most common usage of this pipeline is to submit one JSON input or a directory of JSON inputs. You can refer to [this document](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md) for details on the AlphaFold3 input format.

To submit every `*.json` file in a directory:
`submit_input --input <input_folder> --output <output_folder> [--time <minutes>] [--mem <gb>] [--cpus <n>] [--gpu_type a100|v100]`
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
- `--max_jobs`: Maximum number of Slurm array tasks; inputs are batched to fit (default: `1990`)
- `--overwrite`: Replace an existing generated `inputs_make_feature`, `inputs_make_complex`, or `inputs_both` directory before writing new inputs
- `--check_only`: Check completion status only
- `--check_only_exact`: Report detailed errors from the Slurm logs, and delete the output directories of failed predictions so they are re-run next time
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
- `--max_jobs`: Maximum number of Slurm array tasks; inputs are batched to fit (default: `1990`)
- `--overwrite`: Replace an existing generated `inputs_make_feature`, `inputs_make_complex`, or `inputs_both` directory before writing new inputs
- `--check_only`: Check completion status only
- `--check_only_exact`: Report detailed errors from the Slurm logs, and delete the output directories of failed predictions so they are re-run next time
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
- `--max_jobs`: Maximum number of Slurm array tasks; inputs are batched to fit (default: `1990`)
- `--overwrite`: Replace an existing generated `inputs_both` directory before writing new inputs
- `--check_only`: Check completion status only
- `--check_only_exact`: Report detailed errors from the Slurm logs, and delete the output directories of failed predictions so they are re-run next time
- `--check_stat`: Print pLDDT, ipTM, and pTM statistics

You can also display parameter descriptions by calling `<command> --help`. To see more examples, check [examples](examples/example.md).


# Acknowledgement
This tool is partially based on former alphafold wrapper by Javier, the repository is [here](https://github.com/strubelab/alphafold), kudos to him for setting up a standard to follow, and instructions he has provided me. 
Much appreciation for DeepMind for providing such a great tool.