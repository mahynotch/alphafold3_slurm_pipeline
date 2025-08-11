#!/bin/bash -l
#SBATCH -N 1
#SBATCH --job-name=AF_install
#SBATCH --output=%x.out
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=v100

module load cuda/12.2 gcc/12.2.0

if [ -n "$1" ]
then
    echo "Creating conda environment..."
    if command -v mamba &> /dev/null
    then
        echo "Using mamba to create the environment..."
        mamba env create -f ./environment.yml --quiet --yes --prefix $1
    else
        echo "Using conda to create the environment..."
        conda env create -f ./environment.yml --quiet --yes --prefix $1
    fi
else
    echo "Conda environment already exists."
fi

conda activate $1

if [ $? -eq 0 ]
then
    echo "Conda environment activated."
else
    echo "Failed to activate conda environment. Exiting..."
    exit 1
fi

python modify_config.py "env: $1"

cd alphafold3/
pip install --no-deps -e . --verbose 

build_data

echo "Installation complete."