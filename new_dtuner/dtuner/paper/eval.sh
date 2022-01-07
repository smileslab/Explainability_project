#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=184320


# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt

#conda activate finetune
source /scratch/users/bfrink/new_env/new_finetune/bin/activate

#module load pytorch-py37-cuda10.2-gcc
#module load tensorflow-py37-cuda10.2-gcc/1.15.3

# And finally run the script
bash evaluate_lm.sh ../data/bf_viggo_partial_enrich_1_rebal/test.json ../output_frink_enrich_1_rebal_DataTuner_No_FC ./task_configs/viggo.json 4 2 0
#bash evaluate_lm.sh ../data/anasr3_viggo_partial_enrich_0_req/test.json ../anasr3_output2_frink_partial_enrich_0_DataTuner_No_FC_req/  ./task_configs/viggo_cg.json 4 2 0

hostname
nvidia-smi

