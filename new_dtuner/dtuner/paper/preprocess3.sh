source ./config.sh

assert_run_dir $PAPER_FOLDER_PATTERN

source /scratch/users/bfrink/new_finetune/bin/activate
module load PyTorch/1.8.0-py390-cuda11
echo "Running the data formatting for the ViGGO dataset"
python3.6 experiments/viggo/preprocess.py --in_folder $VIGGO_DATA_LOCATION --out_folder $DATA_FOLDER/bk2_viggo_partial_enrich_1 --classification_dir $DATA_FOLDER/bk2_viggo_partial_enrich_1_consistency


