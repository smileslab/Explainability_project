#!/bin/bash
source ./config.sh

DATASET=viggo
SYSTEM=DataTuner_No_FC_No_FS #DataTuner_No_FC_No_FS #DataTuner_No_FC
OUTPUT_FOLDER=../output_frink_enrich_3key_1_${SYSTEM}
#OUTPUT_FOLDER=../output_frink_answer_3key_1_${SYSTEM}
NUM_PARALLEL=4
NUM_EPOCH=20

if [ -z "$NUM_PARALLEL" ]; then
    NUM_PARALLEL=1
fi

SUFFIX=""
if [[ "$SYSTEM" = "DataTuner_No_FC_No_FS" ]]; then
    SUFFIX="_cg"
fi

echo "Training the model for the dataset $DATASET and writing the trained model to $OUTPUT_FOLDER"

$python -m torch.distributed.launch --nproc_per_node=$NUM_PARALLEL ../src/datatuner/lm/train.py  \
--retrain_base=./lm_training_args/$DATASET/${SYSTEM}_model_training_args.json \
--logdir=$OUTPUT_FOLDER  \
--n_epochs=$NUM_EPOCH \
--patience=5 \
--dataset_path=../data/bf_enrich_3key_1 \
--task_config=./task_configs/${DATASET}${SUFFIX}.json \
--ignore_cache \
--overwrite_output_dir
