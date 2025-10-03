#!/bin/bash

FAIRSEQ_ROOT="../fairseq"
AVHUBERT_ROOT="../av_hubert"

FILE=$(realpath "$0" | sed 's|/run/|/script/|g' | sed 's/\.sh$/.py/')

trap "echo 'Caught Ctrl+C, stopping all processes...'; kill 0; exit 1" SIGINT SIGTERM;

for DEVICE_ID in {0..7}; do

    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${DEVICE_ID} \
    PYTHONPATH=src:${FAIRSEQ_ROOT} \
    python $FILE \
    --nshard 8 \
    --rank ${DEVICE_ID} \
    --user_dir=${AVHUBERT_ROOT}/avhubert \
    &

done

wait
