# Alphafold3 Slurm Pipeline
This repository contains a pipeline for running large scale screening AlphaFold3 on a Slurm cluster. Some concepts are based on the [AlphaFold2 pipeline](https://github.com/strubelab/alphafold). It currently provides the following features:
- Make features: Generate features for two list of protein sequences.
- Make complexes: Generate dimer complexes for two list of protein sequences, based on features.
- Make monomer: Generate monomer models for a list of protein sequences.

To install it, please follow the instructions below (**This installation guide is for Ibex system of KAUST only. For non-Ibex users, please refer to the [Non-ibex User]() part.**).
## Installation
1. Clone this repository.
2. `cd alphafold3_slurm_pipeline`
3. Run `./af3_install.sh` to install the required dependencies. **This process would submit a sbatch job, and could take fair amount of time.**
4. If the installation is failed, you can refer to "AF_install.out" for any error messages. Please contact the author if you have any questions.

## Non-ibex User
For non-Ibex users, you need to make several modifications based on the slurm system you use. The following steps are required:
1. Modify alphafold3_slurm/config.py to set the correct paths for your system.
2. Modify `module load cuda/12.2 gcc/12.2.0` part of the "af3_install.slurm" file to load the correct modules for your system. Usually modern systems should have CUDA and GCC installed.
3. Modify the `--constraint` and `--gres` arguments according to the design of your system.
4. You should be able to run the installation script as above after these modifications.
