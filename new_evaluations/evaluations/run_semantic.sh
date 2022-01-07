#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=demo_ans.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=184320


# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

# Load the module environment suitable for the job
module load CUDA/10.2.2
module load tensorflow2-py37-cuda10.2-gcc/2.2.0

python3 semantic_sim.py
