#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <run_id> <data> [batch_size] [stride]"
    echo "Example: $0 rq6ds8rl allen_vc_2019_vis_probes_transductive 128 5"
    exit 1
fi

RUN_ID=$1
DATA=$2
BATCH_SIZE=${3:-128}
STRIDE=${4:-1}
CKPT_ROOT=../ckpt

echo "Configuration:"
echo "Run ID: $RUN_ID"
echo "Data: $DATA"
echo "Batch Size: $BATCH_SIZE"
echo "Stride: $STRIDE"
echo "Checkpoint Root: $CKPT_ROOT"
echo "-------------------"

echo "Found checkpoints:"
ls ${CKPT_ROOT}/${RUN_ID}/epoch_*.pt | sort -Vr
echo "-------------------"

counter=0
for CKPT in $(ls ${CKPT_ROOT}/${RUN_ID}/epoch_*.pt | sort -Vr); do
    if [ $((counter % STRIDE)) -eq 0 ]; then
        echo "Processing checkpoint: $CKPT"
        python train.py --config-name forward \
           ckpt.load_from=$CKPT \
           data=$DATA \
           batch_size=$BATCH_SIZE \
           num_workers=7
    fi
    counter=$((counter + 1))
done

