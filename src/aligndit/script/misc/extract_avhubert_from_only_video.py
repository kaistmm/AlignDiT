import argparse
import glob
import logging
import math
import os
import sys

import fairseq
import numpy as np
import torch
import tqdm


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
        video_fn, audio_fn = mix_name
        video_feats = self.load_image(video_fn)
        return video_feats

    def load_image(self, audio_name):
        feats = self.custom_utils.load_video(audio_name)
        feats = self.transform(feats)
        feats = np.expand_dims(feats, axis=-1)
        return feats

    def get_feats(self, path, ref_len=None):
        video_feats = self.load_feature(path, ref_len)
        with torch.no_grad():
            video_feats = torch.from_numpy(video_feats.astype(np.float32)).cuda()
            video_feats = video_feats.unsqueeze(dim=0).permute((0, 4, 1, 2, 3)).contiguous()
            source = {"audio": None, "video": video_feats}
            feat, _ = self.model.extract_finetune(
                source=source,
                padding_mask=None,
                mask=False,
            )
            return feat.squeeze(dim=0)


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


def main(args, custom_utils=None):
    nshard = args.nshard
    rank = args.rank
    input_dir = args.input_dir
    output_dir = args.output_dir
    file_extension = args.file_extension

    ckpt_path = args.ckpt_path
    max_chunk = args.max_chunk
    reader = AVHubertFeatureReader(ckpt_path, max_chunk, custom_utils=custom_utils)

    generator, num = get_path_iterator(input_dir, nshard, rank, file_extension)
    iterator = generator()

    for path in tqdm.tqdm(iterator, total=num):
        feature = reader.get_feats((path, None))

        assert input_dir in path
        save_path = path.replace(input_dir, output_dir)
        assert save_path != path
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
        "--input-dir",
        type=str,
        default="data/LRS3_debug/autoavsr/video",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/LRS3_debug/autoavsr/avhubert_video_feat",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="ckpts/av_hubert/lrs3_vox/clean-pretrain/large_vox_iter5.pt",
    )
    parser.add_argument("--file-extension", type=str, default=".mp4", help="Video file extension.")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--user_dir", type=str, default=None)
    args = parser.parse_args()

    fairseq.utils.import_user_module(args)
    sys.path.append(args.user_dir)

    from aligndit.script.misc import avhubert_utils

    logger.info(args)
    main(args, custom_utils=avhubert_utils)
