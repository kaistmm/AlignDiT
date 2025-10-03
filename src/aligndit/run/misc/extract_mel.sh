#!/bin/bash

FILE=$(realpath "$0" | sed 's|/run/|/script/|g' | sed 's/\.sh$/.py/')

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
    for RANK in {0..15}; do

        OMP_NUM_THREADS=1 \
        PYTHONPATH=src \
        python $FILE \
        --nshard 16 \
        --rank ${RANK} \
        &

    done
wait

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
    for RANK in {0..15}; do

        OMP_NUM_THREADS=1 \
        PYTHONPATH=src \
        python $FILE \
        --nshard 16 \
        --rank ${RANK} \
        --input-dir "data/LRS3_debug/autoavsr/audio" \
        --output-dir "data/LRS3_debug/autoavsr/mel_tacotron" \
        --file-extension ".wav" \
        &

    done
wait
