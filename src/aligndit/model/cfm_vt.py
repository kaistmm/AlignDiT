"""
ein notation:
b - batch
n - sequence
nt - text sequence
nv - video sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.cfm import CFM
from f5_tts.model.utils import (
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFM_VT(CFM):
    def __init__(
        self,
        *args,
        text_drop_prob=0.2,
        video_drop_prob=0.2,
        audio_video_ratio=4,
        ctc_lambda=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # multimodal classifier-free guidance
        self.text_drop_prob = text_drop_prob
        self.video_drop_prob = video_drop_prob

        self.audio_video_ratio = audio_video_ratio  # audio to video ratio in length

        # ctc loss
        self.ctc_lambda = ctc_lambda

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        video: float["b nv d"],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        cfg_strength_v=1.0,
        ignore_modality=None,  # "text" | "video" | None
    ):
        self.eval()
        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)
        video = video.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        duration = torch.as_tensor(duration, device=device, dtype=torch.long)
        if duration.ndim == 0:
            duration = duration.expand(batch)

        text_lens = (text != -1).sum(dim=-1)
        duration = torch.maximum(
            torch.maximum(text_lens, lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
            video_mask = mask.repeat_interleave(self.audio_video_ratio, dim=-1)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None
            video_mask = None
        text_mask = lens_to_mask(text_lens)

        # complementary masking
        complementary_mask = cond_mask.squeeze(-1)[:, :: self.audio_video_ratio]
        if exists(video_mask):
            complementary_mask &= video_mask

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5 and cfg_strength_v < 1e-5:
                pred, _ = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    video=video,
                    time=t,
                    mask=mask,
                    text_mask=text_mask,
                    video_mask=video_mask,
                    complementary_mask=complementary_mask,
                    drop_text=ignore_modality == "text",
                    drop_video=ignore_modality == "video",
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for multimodal classifier-free guidance
            pred_cfg, _ = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                video=video,
                time=t,
                mask=mask,
                text_mask=text_mask,
                video_mask=video_mask,
                complementary_mask=complementary_mask,
                drop_text=ignore_modality == "text",
                drop_video=ignore_modality == "video",
                cfg_infer=True,
                cache=True,
            )
            if ignore_modality is None:
                pred, tts_pred, null_pred = torch.chunk(pred_cfg, 3, dim=0)
                return pred + (pred - tts_pred) * cfg_strength_v + (tts_pred - null_pred) * cfg_strength
            elif ignore_modality == "text":  # VTS
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                return pred + (pred - null_pred) * cfg_strength_v
            elif ignore_modality == "video":  # TTS
                pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
                return pred + (pred - null_pred) * cfg_strength
            else:
                raise ValueError(f"ignore_modality {ignore_modality} not supported.")

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        video: float["b nv d"],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        text_lens: int["b"] | None = None,  # noqa: F821
        video_lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        if not exists(text_lens):
            text_lens = torch.full((batch,), text.size(1), device=device)
        text_mask = lens_to_mask(text_lens)

        if not exists(video_lens):
            video_lens = torch.full((batch,), video.size(1), device=device)
        video_mask = lens_to_mask(video_lens)

        if not torch.equal(lens, video_lens * self.audio_video_ratio):
            print(
                f"Warning: lens and video_lens are inconsistent with audio_video_ratio={self.audio_video_ratio}.\n"
                f"lens: {lens.tolist()}\n"
                f"video_lens*{self.audio_video_ratio}: {(video_lens * self.audio_video_ratio).tolist()}"
            )

        # get a random span to mask out for training conditionally
        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_video_mask = mask_from_frac_lengths(video_lens, frac_lengths)
        rand_span_mask = rand_span_video_mask.repeat_interleave(self.audio_video_ratio, dim=-1)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # complementary masking
        complementary_mask = ~rand_span_mask[:, :: self.audio_video_ratio]
        if exists(video_mask):
            complementary_mask &= video_mask

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        drop_text = False
        drop_video = False

        if self.cond_drop_prob + self.text_drop_prob + self.video_drop_prob >= 1.0:
            print(
                f"Warning: Drop probabilities sum to more than 1.0: "
                f"{self.cond_drop_prob + self.text_drop_prob + self.video_drop_prob:.2f}"
            )

        rand_val = random()
        if rand_val < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
            drop_video = True
        elif rand_val < self.cond_drop_prob + self.text_drop_prob:  # for VTS
            drop_text = True
        elif rand_val < self.cond_drop_prob + self.text_drop_prob + self.video_drop_prob:  # for TTS
            drop_video = True

        # apply mask will use more memory; might adjust batchsize or batchsampler long sequence threshold
        pred, intermediates_ctc = self.transformer(
            x=φ,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
            video=video,
            drop_video=drop_video,
            text_mask=text_mask,
            video_mask=video_mask,
            complementary_mask=complementary_mask,
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]
        loss = loss.mean()
        component_losses = {"diff_loss": loss.item()}

        # ctc loss
        if self.ctc_lambda > 0:
            ctc_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
            for intermediate in intermediates_ctc.values():
                z_tilde = intermediate["z_tilde"]
                z_lens = intermediate["z_lens"]
                ctc_loss += F.ctc_loss(
                    z_tilde.transpose(1, 0).log_softmax(-1),  # Log probabilities (n b c)
                    text,  # Target sequences (b nt)
                    z_lens,  # Length of inputs
                    text_lens,  # Length of targets
                    blank=self.transformer.text_embed.text_embed.num_embeddings,  # Blank token index
                    reduction="mean",  # Reduction method ('none', 'mean', 'sum')
                    zero_infinity=True,  # Ignore loss if log(0) happens
                )
            ctc_loss /= len(intermediates_ctc)
            loss += ctc_loss * self.ctc_lambda
            component_losses["ctc_loss"] = ctc_loss.item()

        return loss, component_losses, cond, pred
