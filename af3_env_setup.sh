#!/bin/bash

source ~/.bashrc

# Install the required packages

if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, installing miniforge..."
    wget https://github.com/conda-forge/miniforge/releases/download/24.11.3-0/Miniforge3-24.11.3-0-Linux-x86_64.sh -O miniforge.sh
    bash miniforge.sh -b -p /ibex/user/$USER/miniforge3
    rm miniforge.sh
    source /ibex/user/$USER/miniforge3/bin/activate
    conda init
else
    echo "Conda is already installed."
fi

# Install the required packages with slurm

if [ ! -d "alphafold3" ]; then
    echo "Cloning AlphaFold3 repository..."
    git clone https://github.com/google-deepmind/alphafold3.git
else
    echo "AlphaFold3 directory already exists."
fi

# Move af3 script to bin directory
echo "#!/bin/usr/env python3" > bin/run_alphafold
cat alphafold3/run_alphafold.py >> bin/run_alphafold
chmod +x bin/run_alphafold

# Configure the environment
read -p "Enter the directory where you want to install the environment (default is ./env): " env_dir
env_dir=${env_dir:-./env}
env_dir=$(realpath "$env_dir")
echo env: $env_dir > './alphafold3_slurm/config.yaml'
# python modify_config.py "env: $env_dir"

read -p "Enter the directory where you store your database (default is /ibex/reference/KSL/alphafold/3.0.0): " db_dir
db_dir=${db_dir:-/ibex/reference/KSL/alphafold/3.0.0}
db_dir=$(realpath "$db_dir")
python modify_config.py "db: $db_dir"

read -p "Enter the !directory! where you store the parameter file (default is ./parameter): " param_dir
param_dir=${param_dir:-./parameter}
param_dir=$(realpath "$param_dir")
python modify_config.py "parameter: $param_dir"

# Create the environment directory
if [ ! -d "$env_dir" ]; then
    echo "Creating environment directory at $env_dir..."
    mkdir -p "$env_dir"
else
    echo "Environment directory already exists at $env_dir."
fi

echo "Will installing environment in $env_dir..."
echo "Installation script will be submitted to the cluster. Please use 'squeue -u $USER' to check the status of the job."
sbatch ./af3_install.slurm $env_dir