#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=184320


# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
#conda activate finetune
source /scratch/users/bfrink/new_env/new_finetune/bin/activate

#module load pytorch-py37-cuda10.2-gcc
#module load tensorflow-py37-cuda10.2-gcc/1.15.3

# And finally run the script
#bash evaluate_lm.sh ../data/bf_enrich7_1/test.json ../output_frink_enrich7_1_DataTuner_No_FC_No_FS ./task_configs/viggo.json 1 1 0
#bash evaluate_lm.sh ../data/bf_answer5_4/test.json ../output_frink_answer5_4_DataTuner_No_FC_No_FS ./task_configs/viggo.json 4 1 0

#bash evaluate_lm.sh ../data/bf_enrich_3key_-1/test.json ../output_frink_enrich_3key_1_DataTuner_No_FC_No_FS ./task_configs/viggo.json 1 1 0
bash evaluate_lm.sh ../data/bf_enrich6_-1/test.json gpt2 ./task_configs/viggo.json 1 1 0
#bash evaluate_lm.sh ../data/bf_answer_3key_1/test.json ../output_frink_answer_3key_1_DataTuner_No_FC_No_FS ./task_configs/viggo.json 1 1 0


hostname
nvidia-smi

