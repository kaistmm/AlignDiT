import argparse
import glob
import logging
import math
import os
import sys

import fairseq
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from python_speech_features import logfbank
from scipy.io import wavfile


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_avhubert_feature")


class AVHubertFeatureReader(object):
    def __init__(self, ckpt_path, max_chunk=1600000, custom_utils=None):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval().cuda()
        self.task = task
        self.max_chunk = max_chunk
        self.stack_order_audio = self.task.cfg.stack_order_audio
        image_crop_size, image_mean, image_std = (
            self.task.cfg.image_crop_size,
            self.task.cfg.image_mean,
            self.task.cfg.image_std,
        )
        self.transform = custom_utils.Compose(
            [
                custom_utils.Normalize(0.0, 255.0),
                custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                custom_utils.Normalize(image_mean, image_std),
            ]
        )

        self.custom_utils = custom_utils
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")
        logger.info(f"Transform: {self.transform}")

    def load_feature(self, mix_name, ref_len=None):
        def stacker(feats, stack_order):
            feat_dim = feats.shape[1]
            if len(feats) % stack_order != 0:
                res = stack_order - len(feats) % stack_order
                res = np.zeros([res, feat_dim]).astype(feats.dtype)
                feats = np.concatenate([feats, res], axis=0)
            feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
            return feats

        video_fn, audio_fn = mix_name
        video_feats = self.load_image(video_fn)

        sample_rate, wav_data = wavfile.read(audio_fn)
        assert sample_rate == 16_000 and len(wav_data.shape) == 1
        audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)
        audio_feats = stacker(audio_feats, self.stack_order_audio)

        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = np.concatenate(
                [audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)]
            )
        elif diff > 0:
            audio_feats = audio_feats[:-diff]
        return video_feats, audio_feats

    def load_image(self, audio_name):
        feats = self.custom_utils.load_video(audio_name)
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def get_feats(self, path, ref_len=None):
        video_feats, audio_feats = self.load_feature(path, ref_len)
        with torch.no_grad():
            audio_feats, video_feats = (
                torch.from_numpy(audio_feats.astype(np.float32)).cuda(),
                torch.from_numpy(video_feats.astype(np.float32)).cuda(),
            )
            if self.task.cfg.normalize:
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
            video_feats = video_feats.unsqueeze(dim=0).permute((0, 4, 1, 2, 3)).contiguous()
            audio_feats = audio_feats.unsqueeze(dim=0).transpose(1, 2)
            source = {"audio": audio_feats, "video": video_feats}
            feat, _ = self.model.extract_finetune(
                source=source,
                padding_mask=None,
                mask=False,
            )
            return feat.squeeze(dim=0)


def get_path_iterator(a_root, v_root, nshard, rank, a_file_extension, v_file_extension):
    lines = sorted(glob.glob(f"{v_root}/**/*{v_file_extension}", recursive=True))

    tot = len(lines)
    shard_size = math.ceil(tot / nshard)
    start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
    assert start < end, "start={start}, end={end}"
    logger.info(f"rank {rank} of {nshard}, process {end - start} ({start}-{end}) out of {tot}")

    lines = lines[start:end]

    def iterate():
        for line in lines:
            video_path = line
            audio_path = line.replace(v_root, a_root).replace(v_file_extension, a_file_extension)
            yield (video_path, audio_path)

    return iterate, len(lines)


def main(args, custom_utils=None):
    nshard = args.nshard
    rank = args.rank
    a_input_dir = args.a_input_dir
    v_input_dir = args.v_input_dir
    output_dir = args.output_dir
    a_file_extension = args.a_file_extension
    v_file_extension = args.v_file_extension

    ckpt_path = args.ckpt_path
    max_chunk = args.max_chunk
    reader = AVHubertFeatureReader(ckpt_path, max_chunk, custom_utils=custom_utils)

    generator, num = get_path_iterator(a_input_dir, v_input_dir, nshard, rank, a_file_extension, v_file_extension)
    iterator = generator()

    for path in tqdm.tqdm(iterator, total=num):
        try:
            feature = reader.get_feats(path)
        except Exception:
            print(f"error in processing {path}")
            continue

        video_path, audio_path = path
        assert v_input_dir in video_path
        save_path = video_path.replace(v_input_dir, output_dir)
        assert save_path != video_path
        save_path = os.path.splitext(save_path)[0] + ".npy"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        np.save(save_path, feature.detach().cpu().numpy())
        del feature

    logger.info("finished successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--v-input-dir",
        type=str,
        default="data/LRS3_debug/autoavsr/video",
    )
    parser.add_argument(
        "--a-input-dir",
        type=str,
        default="data/LRS3_debug/autoavsr/audio",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/LRS3_debug/autoavsr/avhubert_feat",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="ckpts/av_hubert/lrs3_vox/clean-pretrain/large_vox_iter5.pt",
    )
    parser.add_argument("--v-file-extension", type=str, default=".mp4", help="Video file extension.")
    parser.add_argument("--a-file-extension", type=str, default=".wav", help="Audio file extension.")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--user_dir", type=str, default=None)
    args = parser.parse_args()

    fairseq.utils.import_user_module(args)
    sys.path.append(args.user_dir)

    from aligndit.script.misc import avhubert_utils

    logger.info(args)
    main(args, custom_utils=avhubert_utils)
