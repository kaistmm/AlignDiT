OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=1200 \
PYTHONPATH=src \
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    src/aligndit/script/train/finetune.py \
    --config-name finetune \
