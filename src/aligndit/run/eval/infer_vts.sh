# For better performance, need to finetune with drop_text=True

OMP_NUM_THREADS=1 \
PYTHONPATH=src \
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    src/aligndit/script/eval/infer.py \
    -n finetune \
    -s 0 -t lrs3_test_cross -nfe 32 -c 400000 \
    --cfg_v 2 --ignore-modality text \
