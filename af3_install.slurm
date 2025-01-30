#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=AF_install
#SBATCH --output=%x.out
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=v100

source ~/.bashrc
module load cuda/12.2 gcc/12.2.0

if ! conda env list | grep -q alphafold3_env
then
    echo "Creating conda environment..."
    if command -v mamba &> /dev/null
    then
        echo "Using mamba to create the environment..."
        mamba env create -f ./environment.yml --quiet --yes
    else
        echo "Using conda to create the environment..."
        conda env create -f ./environment.yml --quiet --yes
    fi
    conda env create -f ./environment.yml --quiet --yes
    conda activate alphafold3_env
else
    echo "Conda environment already exists."
fi

if ! conda env list | grep -q alphafold3_env
then
    echo "Failed to create conda environment. Exiting..."
    exit 1
fi

cd alphafold3/
pip install --no-deps . --verbose 

build_data

echo "Installation complete."

rm AF_install.out