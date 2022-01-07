\source ./config.sh

assert_run_dir $PAPER_FOLDER_PATTERN

source /scratch/users/bfrink/new_env/new_finetune/bin/activate
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cm/shared/apps/Python/3.7.3/lib/
module load pytorch-py37-cuda10.2-gcc 


echo "Running the data formatting for the ViGGO dataset"
#pip install pandas
export PYTHONPATH=$PYTHONPATH:/scratch/users/bfrink/new_dtuner/dtuner/src:/scratch/users/bfrink/new_env/new_finetune/bin/python
#python3.7 experiments/viggo/preprocess.py --in_folder ../bf-viggo-answer5-4 --out_folder $DATA_FOLDER/bf_answer5_4 --classification_dir $DATA_FOLDER/bf_answer5_4_consistency_req
#python3.7 experiments/viggo/preprocess.py --in_folder ../bf-viggo-enrich7-1 --out_folder $DATA_FOLDER/bf_enrich7_1 --classification_dir $DATA_FOLDER/bf_enrich7_1_consistency_req
python3.7 experiments/viggo/preprocess.py --in_folder ../bf-viggo-enrich-3key-1 --out_folder $DATA_FOLDER/bf_enrich_3key_1 --classification_dir $DATA_FOLDER/bf_enrich_3key_1_consistency_req
#python3.7 experiments/viggo/preprocess.py --in_folder ../bf-viggo-answer-3key-1 --out_folder $DATA_FOLDER/bf_answer_3key_1 --classification_dir $DATA_FOLDER/bf_answer_3key_1_$consistency_req
