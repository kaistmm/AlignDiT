#!/bin/bash

FILE=$(realpath "$0" | sed 's|/run/|/script/|g' | sed 's/\.sh$/.py/')

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
    for DEVICE_ID in {0..7}; do

        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${DEVICE_ID} \
        python $FILE \
        --nshard 8 \
        --rank ${DEVICE_ID} \
        &

    done
wait
