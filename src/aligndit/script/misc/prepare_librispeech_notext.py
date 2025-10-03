import os
import sys


sys.path.append(os.getcwd())

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def deal_with_audio_dir(audio_dir):
    sub_result, durations = [], []
    audio_lists = list(audio_dir.rglob("*.flac"))
    for line in audio_lists:
        duration = sf.info(line).duration
        if duration < 0.4 or duration > 30:
            continue
        sub_result.append({"audio_path": str(line), "text": "", "duration": duration})
        durations.append(duration)
    return sub_result, durations


def main():
    result = []
    duration_list = []

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []

    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)
        writer.finalize()

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and save LibriSpeech dataset")
    parser.add_argument(
        "--dataset_dir", type=str, default="data/LibriSpeech_debug/audio", help="Path to LibriSpeech dataset"
    )

    args = parser.parse_args()

    max_workers = 36

    SUB_SET = ["train-clean-100", "train-clean-360", "train-other-500"]
    dataset_dir = args.dataset_dir
    dataset_name = "LibriSpeech_notext"
    save_dir = str(files("aligndit").joinpath("../../")) + f"/data/{dataset_name}"
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}\n")
    main()
