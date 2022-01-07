#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --mem=184320


# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
#source ./venv/bin/activate
# Load the module environment suitable for the job
#module load tensorflow2-py37-cuda10.2-gcc
#pytorch-py37-cuda10.2-gcc/1.5.1 
# module load ml-pythondeps-py37-cuda10.2-gcc
# module unload ml-pythondeps-py37-cuda10.2-gcc/4.0.8
#module load CUDA/10.0.1
#module load Pytorch/1.8.0-py387-cuda11
#module load TensorFlow
#Activate the virtual environment
#source ./venv/bin/activate

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt


# source $HOME/miniconda3/bin/activate
#conda activate finetune
source /scratch/users/bfrink/new_env/new_finetune/bin/activate

module load pytorch-py37-cuda10.2-gcc
module load tensorflow-py37-cuda10.2-gcc/1.15.3 

# And finally run the python script
bash hamza_train_lm.sh

hostname
nvidia-smi
