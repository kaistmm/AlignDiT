import os
import sys


sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from aligndit.model import CFM_VT
from aligndit.model.modules import MelSpec_tacotron
from aligndit.script.eval.utils import get_inference_prompt_vt, get_lrs3_test_metainfo, load_vocoder
from f5_tts.infer.utils_infer import load_checkpoint
from f5_tts.model.utils import get_tokenizer


accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"


use_ema = True
target_rms = 0.1


rel_path = str(files("aligndit").joinpath("../../"))


def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-n", "--expname", required=True)
    parser.add_argument("-c", "--ckptstep", default=1250000, type=int)

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument("-t", "--testset", required=True)

    parser.add_argument("--cfg_t", default=5.0, type=float)
    parser.add_argument("--cfg_v", default=2.0, type=float)
    parser.add_argument("--ignore-modality", default=None, type=str)

    args = parser.parse_args()

    seed = args.seed
    exp_name = args.expname
    ckpt_step = args.ckptstep

    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling

    testset = args.testset

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    ignore_modality = args.ignore_modality
    cfg_strength_t = args.cfg_t
    cfg_strength_v = args.cfg_v
    use_truth_duration = True
    no_ref_audio = False

    model_cfg = OmegaConf.load(str(files("aligndit").joinpath(f"config/{exp_name}.yaml")))
    model_cls = get_class(f"aligndit.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft

    if testset == "lrs3_test_cross":
        metalst = rel_path + "/data/lrs3_test_cross_sentence.lst"
        lrs3_test_path = rel_path + "/data/LRS3_debug/autoavsr"
        metainfo = get_lrs3_test_metainfo(metalst, lrs3_test_path)
    elif testset == "lrs3_test_cross_w_lipreader":
        metalst = rel_path + "/data/lrs3_test_cross_sentence.lst"
        lrs3_test_path = rel_path + "/data/LRS3_debug/autoavsr"
        metainfo = get_lrs3_test_metainfo(metalst, lrs3_test_path, use_lipreader=True)
    else:
        raise ValueError(f"testset {testset} not supported.")

    if ignore_modality is None:
        modality_str = f"_cfgt{cfg_strength_t}_cfgv{cfg_strength_v}"
    elif ignore_modality == "text":
        modality_str = f"_vts_cfg{cfg_strength_v}"
    elif ignore_modality == "video":
        modality_str = f"_tts_cfg{cfg_strength_t}"
    else:
        raise ValueError("ignore_modality should be one of None, 'text', or 'video'.")

    # path to save genereted wavs
    output_dir = (
        f"{rel_path}/"
        f"results/{exp_name}_{ckpt_step}/{testset}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"{modality_str}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )

    # -------------------------------------------------#

    # Vocoder model
    local = True
    if mel_spec_type == "hifigan_16k":
        vocoder_local_path = "ckpts/hifigan_16k_LRS3/g_01000000"
    else:
        raise ValueError("only 'hifigan_16k' vocoder is supported now.")
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path, device=device)

    # Tokenizer
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model
    model = CFM_VT(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_module=MelSpec_tacotron(**model_cfg.model.mel_spec),
        mel_spec_kwargs={k: v for k, v in model_cfg.model.mel_spec.items() if k != "mel_spec_type"},  # hack
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    ckpt_prefix = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}"
    if os.path.exists(ckpt_prefix + ".pt"):
        ckpt_path = ckpt_prefix + ".pt"
    elif os.path.exists(ckpt_prefix + ".safetensors"):
        ckpt_path = ckpt_prefix + ".safetensors"
    else:
        print("Loading from self-organized training checkpoints rather than released pretrained.")
        ckpt_path = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{ckpt_step}.pt"

    dtype = torch.float32
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    prompts_all = get_inference_prompt_vt(
        metainfo,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        mel_spec_type=mel_spec_type,
        target_rms=target_rms,
        use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
        min_secs=1,
        audio_video_ratio=model.audio_video_ratio,
    )

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list, total_videos = prompt
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)
            total_videos = total_videos.to(device)

            # Inference
            with torch.inference_mode():
                generated, _ = model.sample(
                    cond=ref_mels,
                    text=final_text_list,
                    duration=total_mel_lens,
                    video=total_videos,
                    lens=ref_mel_lens,
                    steps=nfe_step,
                    cfg_strength=cfg_strength_t,
                    sway_sampling_coef=sway_sampling_coef,
                    no_ref_audio=no_ref_audio,
                    seed=seed,
                    cfg_strength_v=cfg_strength_v,
                    ignore_modality=ignore_modality,
                )
                # Final result
                for i, gen in enumerate(generated):
                    gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)

                    assert mel_spec_type == "hifigan_16k"
                    generated_wave = vocoder(gen_mel_spec).squeeze(1).cpu()

                    if ref_rms_list[i] < target_rms:
                        generated_wave = generated_wave * ref_rms_list[i] / target_rms
                    save_path = f"{output_dir}/{utts[i]}.wav"
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torchaudio.save(save_path, generated_wave, target_sample_rate)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
