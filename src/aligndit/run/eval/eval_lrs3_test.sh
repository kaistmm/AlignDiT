#!/bin/bash

EVAL_DIRS=(
    # AlignDiT
    "results/finetune_400000/lrs3_test_cross/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
    # AlignDiT with expert lip reader for VTS task
    "results/finetune_400000/lrs3_test_cross_w_lipreader/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
)

for GEN_WAV_DIR in "${EVAL_DIRS[@]}"; do
    echo "========================================================================"
    echo "Evaluating directory: ${GEN_WAV_DIR}"
    echo "========================================================================"

    for METRIC in wer sim; do
        OMP_NUM_THREADS=1 \
        PYTHONPATH=src \
        python src/aligndit/script/eval/eval_lrs3_test.py \
            -e "${METRIC}" \
            -g "${GEN_WAV_DIR}" \
            -n 8
    done
done
