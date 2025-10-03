import argparse
import glob
import logging
import math
import os
import sys

import numpy as np
import torchaudio
from tqdm import tqdm

from gslm.unit2speech.tacotron2.layers import TacotronSTFT


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_mel_spectrogram")


def get_tacotron_mel_spectrogram_processor(
    n_fft=640,
    n_mel_channels=80,
    target_sample_rate=16000,
    hop_length=160,
    win_length=640,
):
    config_mel = {
        "filter_length": n_fft,
        "hop_length": hop_length,
        "win_length": win_length,
        "n_mel_channels": n_mel_channels,
        "sampling_rate": target_sample_rate,
        "mel_fmin": 0.0,
        "mel_fmax": target_sample_rate / 2.0,
    }
    processor = TacotronSTFT(**config_mel)
    return processor


def single_job(mel_spec_processor, path, sample_rate):
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    if len(wav.shape) == 3:
        wav = wav.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(wav.shape) == 2

    mel_spec = mel_spec_processor.mel_spectrogram(wav).squeeze(0).transpose(0, 1)
    return mel_spec


def main(args):
    nshard = args.nshard
    rank = args.rank
    input_dir = args.input_dir
    output_dir = args.output_dir
    sample_rate = args.sample_rate
    file_extension = args.file_extension

    mel_spec_processor = get_tacotron_mel_spectrogram_processor()

    generator, num = get_path_iterator(input_dir, nshard, rank, file_extension)
    iterator = generator()

    for path in tqdm(iterator):
        feature = single_job(mel_spec_processor, path, sample_rate)

        assert input_dir in path
        save_path = path.replace(input_dir, output_dir)
        assert save_path != path
        save_path = os.path.splitext(save_path)[0] + ".npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        assert len(feature.shape) == 2
        np.save(save_path, feature.detach().cpu().numpy())
        del feature


def get_path_iterator(root, nshard, rank, file_extension):
    lines = sorted(glob.glob(f"{root}/**/*{file_extension}", recursive=True))

    tot = len(lines)
    shard_size = math.ceil(tot / nshard)
    start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
    assert start < end, "start={start}, end={end}"
    logger.info(f"rank {rank} of {nshard}, process {end-start} " f"({start}-{end}) out of {tot}")

    lines = lines[start:end]

    def iterate():
        for line in lines:
            yield line

    return iterate, len(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/LibriSpeech_debug/audio",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/LibriSpeech_debug/mel_tacotron",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for audio processing.")
    parser.add_argument(
        "--file-extension", type=str, default=".flac", help="Audio file extension (e.g., .wav, .flac, .mp3)."
    )
    args = parser.parse_args()

    main(args)
