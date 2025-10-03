# Evaluate with LRS3 test cross

import argparse
import json
import os
import sys


sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files

import numpy as np
from jiwer import compute_measures

from aligndit.script.eval.utils import get_lrs3_test, run_asr_wer
from f5_tts.eval.utils_eval import run_sim


rel_path = str(files("aligndit").joinpath("../../"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wer"])
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-n", "--gpu_nums", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()


def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    gen_wav_dir = args.gen_wav_dir
    metalst = rel_path + "/data/lrs3_test_cross_sentence.lst"
    lrs3_test_path = rel_path + "/data/LRS3_debug/autoavsr"

    gpus = list(range(args.gpu_nums))
    test_set = get_lrs3_test(metalst, gen_wav_dir, gpus, lrs3_test_path)

    local = args.local
    if local:  # use local custom checkpoint dir
        asr_ckpt_dir = "../checkpoints/Systran/faster-whisper-large-v3"
    else:
        asr_ckpt_dir = ""  # auto download to cache dir
    wavlm_ckpt_dir = "../checkpoints/UniSpeech/wavlm_large_finetune.pth"

    # --------------------------------------------------------------------------

    full_results = []
    metrics = []

    result_path = f"{gen_wav_dir}/_{eval_task}_results.jsonl"

    # We use micro averaging for WER
    if eval_task == "wer":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer, args)
            for r in results:
                full_results.extend(r)

        refs = [r["truth"] for r in full_results]
        hypos = [r["hypo"] for r in full_results]
        metric = compute_measures(refs, hypos)["wer"]
        with open(result_path, "w") as f:
            for line in full_results:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            metric = round(metric, 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")

    elif eval_task == "sim":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, args)
            for r in results:
                full_results.extend(r)

        with open(result_path, "w") as f:
            for line in full_results:
                metrics.append(line[eval_task])
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            metric = round(np.mean(metrics), 5)
            f.write(f"\n{eval_task.upper()}: {metric}\n")

    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    main()
