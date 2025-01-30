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

echo "#!/bin/usr/env python3" > bin/run_alphafold
cat alphafold3/run_alphafold.py >> bin/run_alphafold

sbatch ./af3_install.slurm