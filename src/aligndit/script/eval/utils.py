import json
import math
import os
import random
import string
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from aligndit.model.dataset import cut_or_pad
from aligndit.model.hifigan_16k import Generator
from aligndit.model.modules import MelSpec_tacotron
from f5_tts.eval.utils_eval import padded_mel_batch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# load vocoder
def load_vocoder(vocoder_name="hifigan_16k", is_local=False, local_path="", device=None, hf_cache_dir=None):
    if vocoder_name == "hifigan_16k":
        assert is_local and local_path
        config_file = os.path.join(os.path.dirname(local_path), "config.json")
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        vocoder = Generator(h).to(device)
        state_dict_g = torch.load(local_path, weights_only=True, map_location="cpu")
        vocoder.load_state_dict(state_dict_g["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
    else:
        raise ValueError("only 'hifigan_16k' vocoder is supported now.")
    return vocoder


# metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_lrs3_test_metainfo(metalst, lrs3_test_path, use_lipreader=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []

    split = "test"
    for line in lines:
        folder_name, ref_id, gen_id = line.strip().split("_")

        ref_txt_name = "text_autoavsr_asr" if use_lipreader else "text"
        ref_txt_path = os.path.join(lrs3_test_path, ref_txt_name, split, folder_name, ref_id + ".txt")
        ref_txt = open(ref_txt_path).readline().strip().lower()

        gen_txt_name = "text_autoavsr_vsr" if use_lipreader else "text"
        gen_txt_path = os.path.join(lrs3_test_path, gen_txt_name, split, folder_name, gen_id + ".txt")
        gen_txt = open(gen_txt_path).readline().strip().lower()

        ref_wav = os.path.join(lrs3_test_path, "audio", split, folder_name, ref_id + ".wav")
        gen_wav = os.path.join(lrs3_test_path, "audio", split, folder_name, gen_id + ".wav")

        gen_utt = os.path.join(split, folder_name, gen_id)

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo


# get prompts from metainfo containing: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_inference_prompt_vt(
    metainfo,
    tokenizer="char",
    target_sample_rate=16000,
    n_fft=640,
    win_length=640,
    n_mel_channels=80,
    hop_length=160,
    mel_spec_type="hifigan_16k",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=3,
    max_secs=40,
    audio_video_ratio=4,
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list, total_videos = (
        [[] for _ in range(num_buckets)] for _ in range(7)
    )

    mel_spectrogram = MelSpec_tacotron(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    for utt, prompt_text, prompt_wav, gt_text, gt_wav in tqdm(metainfo, desc="Processing prompts..."):
        # Audio
        ref_audio, ref_sr = torchaudio.load(prompt_wav)
        ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
        if ref_rms < target_rms:
            ref_audio = ref_audio * target_rms / ref_rms
        assert ref_audio.shape[-1] > 5000, f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
        if ref_sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
            ref_audio = resampler(ref_audio)

        # Text
        if len(prompt_text[-1].encode("utf-8")) == 1:
            prompt_text = prompt_text + " "
        text = [prompt_text + gt_text]
        assert tokenizer == "char", "Only char tokenizer is supported."
        text_list = text

        # to mel spectrogram
        ref_mel_path = os.path.splitext(prompt_wav.replace("/audio/", "/mel_tacotron/"))[0] + ".npy"
        if os.path.exists(ref_mel_path):
            ref_mel = torch.from_numpy(np.load(ref_mel_path).T)
        else:
            ref_mel = mel_spectrogram(ref_audio)
            ref_mel = ref_mel.squeeze(0)

        # Duration, mel frame length
        ref_mel_len = ref_mel.shape[-1]

        if use_truth_duration:
            gt_audio, gt_sr = torchaudio.load(gt_wav)
            if gt_sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(gt_sr, target_sample_rate)
                gt_audio = resampler(gt_audio)
            total_mel_len = ref_mel_len + int(gt_audio.shape[-1] / hop_length)

            # # test vocoder resynthesis
            # ref_audio = gt_audio
        else:
            ref_text_len = len(prompt_text.encode("utf-8"))
            gen_text_len = len(gt_text.encode("utf-8"))
            total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len)

        ref_video_path = os.path.splitext(prompt_wav.replace("/audio/", "/avhubert_video_feat/"))[0] + ".npy"
        ref_video = torch.from_numpy(np.load(ref_video_path))
        gt_video_path = os.path.splitext(gt_wav.replace("/audio/", "/avhubert_video_feat/"))[0] + ".npy"
        gt_video = torch.from_numpy(np.load(gt_video_path))

        ref_mel_len = len(ref_video) * audio_video_ratio
        ref_mel = cut_or_pad(ref_mel, ref_mel_len, dim=1, mode="replicate")
        total_mel_len = ref_mel_len + len(gt_video) * audio_video_ratio

        total_video = torch.cat(
            [
                torch.zeros_like(ref_video),
                gt_video,
            ]
        )

        # deal with batch
        assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
        assert (
            min_tokens <= total_mel_len <= max_tokens
        ), f"Audio {utt} has duration {total_mel_len * hop_length // target_sample_rate}s out of range [{min_secs}, {max_secs}]."
        bucket_i = math.floor((total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets)

        utts[bucket_i].append(utt)
        ref_rms_list[bucket_i].append(ref_rms)
        ref_mels[bucket_i].append(ref_mel)
        ref_mel_lens[bucket_i].append(ref_mel_len)
        total_mel_lens[bucket_i].append(total_mel_len)
        final_text_list[bucket_i].extend(text_list)
        total_videos[bucket_i].append(total_video)

        batch_accum[bucket_i] += total_mel_len

        if batch_accum[bucket_i] >= infer_batch_size:
            # print(f"\n{len(ref_mels[bucket_i][0][0])}\n{ref_mel_lens[bucket_i]}\n{total_mel_lens[bucket_i]}")
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                    torch.stack(total_videos[bucket_i]),
                )
            )
            batch_accum[bucket_i] = 0
            (
                utts[bucket_i],
                ref_rms_list[bucket_i],
                ref_mels[bucket_i],
                ref_mel_lens[bucket_i],
                total_mel_lens[bucket_i],
                final_text_list[bucket_i],
                total_videos[bucket_i],
            ) = [], [], [], [], [], [], []

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                    torch.stack(total_videos[bucket_i]),
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)

    return prompts_all


# get lrs3 test-clean cross sentence test
def get_lrs3_test(metalst, gen_wav_dir, gpus, lrs3_test_path, eval_ground_truth=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        folder_name, ref_id, gen_id = line.strip().split("_")
        ref_txt_path = os.path.join(lrs3_test_path, "text", "test", folder_name, ref_id + ".txt")
        ref_txt = open(ref_txt_path).readline().strip().lower()

        gen_txt_path = os.path.join(lrs3_test_path, "text", "test", folder_name, gen_id + ".txt")
        gen_txt = open(gen_txt_path).readline().strip().lower()

        ref_utt = os.path.join("test", folder_name, ref_id)
        gen_utt = os.path.join("test", folder_name, gen_id)

        if eval_ground_truth:
            gen_wav = os.path.join(lrs3_test_path, "audio", gen_utt + ".wav")
        else:
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")

        ref_wav = os.path.join(lrs3_test_path, "audio", ref_utt + ".wav")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# load asr model
def load_asr_model(lang, rank, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device_index=rank, compute_type="float16")
    return model


# WER Evaluation, the way Seed-TTS does
def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        pass
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    asr_model = load_asr_model(lang, rank, ckpt_dir=ckpt_dir)

    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation
    wer_results = []

    from jiwer import compute_measures

    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
            hypo = ""
            for segment in segments:
                hypo = hypo + " " + segment.text

        raw_truth = truth
        raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()

        measures = compute_measures(truth, hypo)
        wer = measures["wer"]

        # ref_list = truth.split(" ")
        # subs = measures["substitutions"] / len(ref_list)
        # dele = measures["deletions"] / len(ref_list)
        # inse = measures["insertions"] / len(ref_list)

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": truth,
                "hypo": hypo,
                "raw_truth": raw_truth,
                "raw_hypo": raw_hypo,
                "wer": wer,
            }
        )

    return wer_results
