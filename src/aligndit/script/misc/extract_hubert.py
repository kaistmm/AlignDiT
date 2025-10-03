import argparse
import glob
import logging
import math
import os
import sys

import numpy as np
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_ssl_feature")


def single_job(feature_extractor, ssl_model, path, sample_rate):
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    input_values = feature_extractor(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
    output = ssl_model(input_values.to(ssl_model.device), output_hidden_states=True)
    rep = output.last_hidden_state  # [B, T, D]
    return rep


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


def main(args):
    nshard = args.nshard
    rank = args.rank
    input_dir = args.input_dir
    output_dir = args.output_dir
    sample_rate = args.sample_rate
    file_extension = args.file_extension

    ssl_model_path = args.ssl_model_path
    feature_extractor = AutoFeatureExtractor.from_pretrained(ssl_model_path)
    ssl_model = AutoModel.from_pretrained(ssl_model_path).eval().cuda()

    generator, num = get_path_iterator(input_dir, nshard, rank, file_extension)
    iterator = generator()

    for path in tqdm(iterator):
        feature = single_job(feature_extractor, ssl_model, path, sample_rate)

        assert input_dir in path
        save_path = path.replace(input_dir, output_dir)
        assert save_path != path
        save_path = os.path.splitext(save_path)[0] + ".npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        feature = feature.squeeze(0)
        assert len(feature.shape) == 2
        np.save(save_path, feature.detach().cpu().numpy())
        del feature


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
        default="data/LibriSpeech_debug/hubert_large_ll60k",
    )
    parser.add_argument(
        "--ssl-model-path",
        type=str,
        default="facebook/hubert-large-ll60k",
        help="Path or HuggingFace identifier for the SSL model (e.g., Hubert, Wav2Vec2).",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate for audio processing.")
    parser.add_argument(
        "--file-extension", type=str, default=".flac", help="Audio file extension (e.g., .wav, .flac, .mp3)."
    )
    args = parser.parse_args()

    logger.info(args)
    main(args)
