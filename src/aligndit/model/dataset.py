import json
import os
from importlib.resources import files

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn

from f5_tts.model.dataset import CustomDataset


def cut_or_pad(data, size, dim=0, mode="constant", value=None):
    """
    Pads or trims the data along a dimension.
    """
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.nn.functional.pad(
            data.unsqueeze(0), [0] * (2 * (data.dim() - dim) - 1) + [padding], mode=mode, value=value
        ).squeeze(0)
        size = data.size(dim)
    elif data.size(dim) > size:
        if dim == 0:
            data = data[:size]
        elif dim == 1:
            data = data[:, :size]
        else:
            assert False
    assert data.size(dim) == size
    return data


class CustomDataset_mel(CustomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data.filter(lambda example: 0.3 <= example["duration"] <= 30)
        self.durations = self.data["duration"]

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except Exception as e:
            print(f"Error in loading data index {index}: {e}. Return None.")
            return None

    def getitem(self, index):
        row = self.data[index]
        audio_path = row["audio_path"]
        text = row["text"]
        mel_path = os.path.splitext(audio_path.replace("/audio/", "/mel_tacotron/"))[0] + ".npy"
        mel_spec = torch.from_numpy(np.load(mel_path).T)
        return {
            "mel_spec": mel_spec,
            "text": text,
            "audio_path": audio_path,
        }

    @staticmethod
    def collate_fn(batch):
        mel_specs = [item["mel_spec"].squeeze(0) for item in batch if item is not None]

        if len(mel_specs) == 0:
            return {}

        mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
        max_mel_length = mel_lengths.amax()

        padded_mel_specs = []
        for spec in mel_specs:  # TODO. maybe records mask for attention here
            padding = (0, max_mel_length - spec.size(-1))
            padded_spec = F.pad(spec, padding, value=0)
            padded_mel_specs.append(padded_spec)

        mel_specs = torch.stack(padded_mel_specs)

        text = [item["text"] for item in batch]
        text_lengths = torch.LongTensor([len(item) for item in text])

        return dict(
            mel=mel_specs,
            mel_lengths=mel_lengths,
            text=text,
            text_lengths=text_lengths,
        )


class CustomDataset_mel_rep(CustomDataset_mel):
    def __getitem__(self, index):
        ret = super().__getitem__(index)
        if ret is not None:
            audio_path = ret["audio_path"]
            rep_path = os.path.splitext(audio_path.replace("/audio/", "/hubert_large_ll60k/"))[0] + ".npy"
            ret["rep"] = torch.from_numpy(np.load(rep_path))
        return ret

    @staticmethod
    def collate_fn(batch):
        ret = CustomDataset_mel.collate_fn(batch)
        if ret == {}:
            return {}

        # rep: [T x D]
        reps = [item["rep"] for item in batch]
        rep_lengths = torch.LongTensor([len(rep) for rep in reps])
        max_rep_length = rep_lengths.amax()

        padded_reps = []
        for rep in reps:  # TODO. maybe records mask for attention here
            padding = (0, 0, 0, max_rep_length - len(rep))
            padded_rep = F.pad(rep, padding, value=0)
            padded_reps.append(padded_rep)

        reps = torch.stack(padded_reps)

        ret["rep"] = reps
        ret["rep_lengths"] = rep_lengths

        return ret


class CustomDataset_mel_video(CustomDataset_mel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_fps = 25
        self.mel_spec_fps = self.target_sample_rate // self.hop_length
        assert self.mel_spec_fps == 100
        self.audio_video_ratio = self.mel_spec_fps // self.video_fps
        assert self.audio_video_ratio == 4

    def __getitem__(self, index):
        ret = super().__getitem__(index)
        if ret is not None:
            mel_spec = ret["mel_spec"]
            audio_path = ret["audio_path"]
            video_path = os.path.splitext(audio_path.replace("/audio/", "/avhubert_video_feat/"))[0] + ".npy"
            video = torch.from_numpy(np.load(video_path))

            mel_spec = cut_or_pad(mel_spec, len(video) * self.audio_video_ratio, dim=1, mode="replicate")
            ret["mel_spec"] = mel_spec
            ret["video"] = video

        return ret

    @staticmethod
    def collate_fn(batch):
        ret = CustomDataset_mel.collate_fn(batch)
        if ret == {}:
            return {}

        video_feats = [item["video"] for item in batch]
        video_lengths = torch.LongTensor([len(feat) for feat in video_feats])
        max_video_length = video_lengths.amax()

        padded_video_feats = []
        for feat in video_feats:  # TODO. maybe records mask for attention here
            video_padding = (0, 0, 0, max_video_length - len(feat))
            padded_feat = F.pad(feat, video_padding, value=0)
            padded_video_feats.append(padded_feat)
        video = torch.stack(padded_video_feats)

        ret["video"] = video
        ret["video_lengths"] = video_lengths

        return ret


# Load dataset


def load_dataset_mel(
    dataset_name: str,
    tokenizer: str = "char",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset | CustomDataset_mel | CustomDataset_mel_rep | CustomDataset_mel_video:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type in ["CustomDataset", "CustomDataset_mel", "CustomDataset_mel_rep", "CustomDataset_mel_video"]:
        if tokenizer:
            rel_data_path = str(files("aligndit").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        else:
            rel_data_path = str(files("aligndit").joinpath(f"../../data/{dataset_name}"))
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]

        if dataset_type == "CustomDataset":
            dataset_cls = CustomDataset
        elif dataset_type == "CustomDataset_mel":
            dataset_cls = CustomDataset_mel
        elif dataset_type == "CustomDataset_mel_rep":
            dataset_cls = CustomDataset_mel_rep
        elif dataset_type == "CustomDataset_mel_video":
            dataset_cls = CustomDataset_mel_video

        train_dataset = dataset_cls(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    return train_dataset
