from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from f5_tts.model.modules import DiTBlock
from f5_tts.model.utils import lens_to_mask
from gslm.unit2speech.tacotron2.layers import TacotronSTFT


def get_tacotron_mel_spectrogram(
    waveform,
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
    processor = TacotronSTFT(**config_mel).to(waveform.device)

    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = processor.mel_spectrogram(waveform).squeeze(0).transpose(0, 1)
    return mel


class MelSpec_tacotron(nn.Module):
    def __init__(
        self,
        n_fft=640,
        hop_length=160,
        win_length=640,
        n_mel_channels=80,
        target_sample_rate=16_000,
        mel_spec_type="hifigan_16k",
    ):
        super().__init__()
        if mel_spec_type != "hifigan_16k":
            raise ValueError("only 'hifigan_16k' mel_spec_type is supported now.")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        self.extractor = get_tacotron_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


class DownsampleLayer(nn.Module):
    def __init__(
        self,
        sampling_ratios: Tuple,
        in_channels: int,
        channels: int,
        out_channels: int,
        groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for i, r in enumerate(sampling_ratios):
                module = nn.Conv1d(in_channels if i == 0 else channels, channels, 3, r, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens):
        # x in (B, T, D)
        # y in (L)
        for r in self.sampling_ratios:
            ylens = (ylens + 2 * 1 - 3) // r + 1
        mask = lens_to_mask(ylens).unsqueeze(-1)
        x = x.transpose(1, 2).contiguous()
        out = self.model(x).transpose(1, 2).contiguous()
        return out * mask, ylens


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        sampling_ratios: Tuple,
        in_channels: int,
        channels: int,
        out_channels: int,
        groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for i, r in enumerate(sampling_ratios):
                module = nn.ConvTranspose1d(in_channels if i == 0 else channels, channels, 4, r, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens):
        # x in (B, T, D)
        # y in (L)
        for r in self.sampling_ratios:
            ylens = (ylens - 1) * r - 2 * 1 + 4
        mask = lens_to_mask(ylens).unsqueeze(-1)
        x = x.transpose(1, 2).contiguous()
        out = self.model(x).transpose(1, 2).contiguous()
        return out * mask, ylens


class DiTCrossBlock(DiTBlock):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" or "flash_attn"
        attn_mask_enabled=True,
        text_dim=512,
    ):
        super().__init__(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            ff_mult=ff_mult,
            dropout=dropout,
            qk_norm=qk_norm,
            pe_attn_head=pe_attn_head,
            attn_backend=attn_backend,
            attn_mask_enabled=attn_mask_enabled,
        )

        if attn_backend != "torch":
            raise NotImplementedError("only torch attn backend is supported for DiTCrossBlock")

        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, kdim=text_dim, vdim=text_dim, batch_first=True
        )

    def forward(self, x, t, mask=None, rope=None, text=None, text_mask=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        # cross attention
        ca_output, _ = self.cross_attn(x, text, text, key_padding_mask=~text_mask, need_weights=False)
        x = x + ca_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x
