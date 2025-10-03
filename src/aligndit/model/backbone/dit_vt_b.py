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

import torch
import torch.nn as nn

from aligndit.model.modules import DownsampleLayer, UpsampleLayer
from cosyvoice.transformer.encoder import ConformerEncoder
from f5_tts.model.backbones.dit import ConvPositionEmbedding, DiT
from f5_tts.model.utils import lens_to_mask


class InputEmbedding_VT_Prefix(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, use_conformer=False):
        super().__init__()
        vid_dim = text_dim
        self.proj = nn.Linear(mel_dim * 2 + vid_dim, out_dim)
        self.text_proj = nn.Linear(text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

        self.register_buffer("vid_null_emb", nn.Parameter(torch.randn(1, 1024) / 1024**0.5))
        self.vid_linear = UpsampleLayer([2, 2], 1024, vid_dim, vid_dim)
        self.use_conformer = use_conformer
        if self.use_conformer:
            self.vid_conformer = ConformerEncoder(
                input_size=vid_dim,
                output_size=vid_dim,
                attention_heads=4,
                linear_units=1024,
                num_blocks=2,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.1,
                normalize_before=True,
                input_layer="linear",
                pos_enc_layer_type="rel_pos_espnet",
                selfattention_layer_type="rel_selfattn",
                use_cnn_module=False,
                macaron_style=False,
            )

    def forward(
        self,
        x: float["b n d"],  # noqa: F722
        cond: float["b n d"],  # noqa: F722
        text_embed: float["b n d"],  # noqa: F722
        video: float["b nv d"],  # noqa: F722
        drop_audio_cond=False,
        drop_video=False,
        text_mask: bool["b nt"] | None = None,  # noqa: F722
        video_mask: bool["b nv"] | None = None,  # noqa: F722
        complementary_mask: bool["b nv"] | None = None,  # noqa: F722
    ):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        if complementary_mask is not None:
            video = torch.where(
                complementary_mask[..., None], self.vid_null_emb.expand(video.size(0), video.size(1), -1), video
            )

        if drop_video:
            video = self.vid_null_emb.expand(video.size(0), video.size(1), -1)

        video_lens = (
            video_mask.sum(dim=1)
            if video_mask is not None
            else torch.full((video.size(0),), video.size(1), device=video.device, dtype=torch.long)
        )

        vid_embed, video_lens = self.vid_linear(video, video_lens)
        if self.use_conformer:
            vid_embed, _ = self.vid_conformer(vid_embed, video_lens)

        x = self.proj(torch.cat((x, cond, vid_embed), dim=-1))

        text_embed = self.text_proj(text_embed)

        if text_mask is not None:
            text_len = text_embed.size(1)
            seq_len = x.size(1)
            x_new = x.new_zeros(x.size(0), text_len + seq_len, x.size(2))
            x_new[:, :text_len][text_mask] = x[:, :text_len][text_mask]

            text_lens = text_mask.sum(dim=1)
            x_idx = text_lens.unsqueeze(1) + torch.arange(seq_len, device=x.device).unsqueeze(0)
            x_new.scatter_(1, x_idx.unsqueeze(-1).expand(-1, -1, x.size(2)), x)
            x = x_new
        else:
            x = torch.cat((text_embed, x), dim=1)

        x = self.conv_pos_embed(x) + x
        return x


class DiT_VT_Prefix(DiT):
    def __init__(
        self,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        use_conformer=True,
        layer_indices_ctc=[6, 12],
        projector_dim=None,
    ):
        super().__init__(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            ff_mult=ff_mult,
            mel_dim=mel_dim,
            text_num_embeds=text_num_embeds,
            text_dim=text_dim,
            text_mask_padding=text_mask_padding,
            text_embedding_average_upsampling=text_embedding_average_upsampling,
            qk_norm=qk_norm,
            conv_layers=conv_layers,
            pe_attn_head=pe_attn_head,
            attn_backend=attn_backend,
            attn_mask_enabled=attn_mask_enabled,
            long_skip_connection=long_skip_connection,
            checkpoint_activations=checkpoint_activations,
        )
        self.input_embed = InputEmbedding_VT_Prefix(mel_dim, text_dim, dim, use_conformer=use_conformer)

        projector_dim = self.dim if projector_dim is None else projector_dim
        z_dim = self.text_embed.text_embed.num_embeddings + 1
        self.layer_map_ctc = {v: i for i, v in enumerate(layer_indices_ctc)}
        self.projectors_ctc = nn.ModuleList(
            [DownsampleLayer([2, 1], self.dim, projector_dim, z_dim) for _ in self.layer_map_ctc]
        )

        # zero init text emb
        nn.init.constant_(self.input_embed.text_proj.weight, 0)
        # zero init video emb
        nn.init.constant_(self.input_embed.proj.weight[:, mel_dim * 2 :], 0)

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        video,  # b nv d
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        drop_video: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,  # noqa: F722
        text_mask: bool["b nt"] | None = None,  # noqa: F722
        video_mask: bool["b nv"] | None = None,  # noqa: F722
        complementary_mask: bool["b nv"] | None = None,  # noqa: F722
    ):
        seq_len = text_mask.sum(dim=1).max().item()
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, seq_len, drop_text=True, audio_mask=audio_mask)
                text_embed = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, seq_len, drop_text=False, audio_mask=audio_mask)
                text_embed = self.text_cond
        else:
            text_embed = self.text_embed(text, seq_len, drop_text=drop_text, audio_mask=audio_mask)

        x = self.input_embed(
            x,
            cond,
            text_embed,
            video,
            drop_audio_cond=drop_audio_cond,
            drop_video=drop_video,
            text_mask=text_mask,
            video_mask=video_mask,
            complementary_mask=complementary_mask,
        )

        return x, text_embed

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        video: float["b n d"],  # video  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        mask: bool["b n"] | None = None,  # noqa: F722
        text_mask: bool["b n"] | None = None,  # noqa: F722
        video_mask: bool["b n"] | None = None,  # noqa: F722
        complementary_mask: bool["b n"] | None = None,  # noqa: F722
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        drop_video: bool = False,  # cfg for video
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)

        embed_kwargs = {
            "x": x,
            "cond": cond,
            "text": text,
            "video": video,
            "cache": cache,
            "audio_mask": mask,
            "text_mask": text_mask,
            "complementary_mask": complementary_mask,
        }

        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_list = []
            if not (drop_text or drop_video):
                x_cond, _ = self.get_input_embed(**embed_kwargs)
                x_list.append(x_cond)
                x_tts, _ = self.get_input_embed(
                    **embed_kwargs,
                    drop_video=True,
                )
                x_list.append(x_tts)
            else:
                x_cond, _ = self.get_input_embed(
                    **embed_kwargs,
                    drop_text=drop_text,
                    drop_video=drop_video,
                )
                x_list.append(x_cond)

            x_uncond, _ = self.get_input_embed(
                **embed_kwargs,
                drop_audio_cond=True,
                drop_text=True,
                drop_video=True,
            )
            x_list.append(x_uncond)

            rep_n = len(x_list)
            x = torch.cat(x_list, dim=0)
            t = t.repeat_interleave(rep_n, dim=0)
            masks_to_repeat = [mask, text_mask, video_mask, complementary_mask]
            (
                mask,
                text_mask,
                video_mask,
                complementary_mask,
            ) = [m.repeat_interleave(rep_n, dim=0) if m is not None else None for m in masks_to_repeat]

        else:
            x, _ = self.get_input_embed(
                **embed_kwargs,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                drop_video=drop_video,
            )

        text_lens = (
            text_mask.sum(dim=1)
            if text_mask is not None
            else torch.full((x.size(0),), text.size(1), device=x.device, dtype=torch.long)
        )
        lens = (
            mask.sum(dim=1)
            if mask is not None
            else torch.full((x.size(0),), seq_len, device=x.device, dtype=torch.long)
        )

        # x_new = torch.zeros_like(x)
        # if text_mask is not None:
        #     x_new[:, :text.size(1)][text_mask] = x[:, :text.size(1)][text_mask]
        # else:
        #     x_new[:, :text.size(1)] = x[:, :text.size(1)]
        x_idx = text_lens.unsqueeze(1) + torch.arange(seq_len, device=x.device).unsqueeze(0)
        # x_new.scatter_(1, x_idx.unsqueeze(-1).expand(-1, -1, x.size(2)), x[:, text.size(1):text.size(1)+seq_len, :])
        # x = x_new

        concat_lens = text_lens + lens
        concat_mask = lens_to_mask(concat_lens)

        assert x.size(1) == text.size(1) + seq_len
        rope = self.rotary_embed.forward_from_seq_len(text.size(1) + seq_len)

        if self.long_skip_connection is not None:
            residual = x

        intermediates_ctc = {}
        for layer_i, block in enumerate(self.transformer_blocks):
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block),
                    x,
                    t,
                    None if self.training else concat_mask,  # memory issue
                    rope,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    t,
                    mask=None if self.training else concat_mask,  # memory issue
                    rope=rope,
                )

            if not cache and layer_i in self.layer_map_ctc:  # hack
                projector = self.projectors_ctc[self.layer_map_ctc[layer_i]]
                x_ = x.gather(1, x_idx.unsqueeze(-1).expand(-1, -1, x.size(2)))
                z_tilde, z_lens = projector(x_, lens)
                intermediates_ctc[layer_i] = {"z_tilde": z_tilde, "z_lens": z_lens}

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = x.gather(1, x_idx.unsqueeze(-1).expand(-1, -1, x.size(2)))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output, intermediates_ctc
