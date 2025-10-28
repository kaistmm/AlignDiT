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

trap "echo 'Caught Ctrl+C, stopping all processes...'; kill 0; exit 1" SIGINT SIGTERM;

EVAL_DIRS=(
    # AlignDiT
    "results/finetune_400000/lrs3_test_cross/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
    # AlignDiT with expert lip reader for VTS task
    "results/finetune_400000/lrs3_test_cross_w_lipreader/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
)
for GEN_WAV_DIR in "${EVAL_DIRS[@]}"; do
    for DEVICE_ID in {0..7}; do

        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${DEVICE_ID} \
        PYTHONPATH=src:${FAIRSEQ_ROOT} \
        python $FILE \
        --nshard 8 \
        --rank ${DEVICE_ID} \
        --a-input-dir ${GEN_WAV_DIR} \
        --output-dir ${GEN_WAV_DIR}/avhubert_feat \
        --user_dir=${AVHUBERT_ROOT}/avhubert \
        &

    done
    wait
done

wait
