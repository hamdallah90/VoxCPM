"""Utility script for LoRA fine-tuning of VoxCPM on Arabic speech datasets.

This script demonstrates a lightweight strategy to adapt VoxCPM to a new
single-speaker corpus (for example, ``arbml/arabic_single_speaker_speech_dataset``)
using Low-Rank Adaptation (LoRA). It focuses on the attention projections inside
MiniCPM as well as the acoustic bridge layers so that only a few million
parameters are trained while the base checkpoint stays frozen.

Usage (minimal example)::

    HF_TOKEN=hf_... \
    python examples/finetune_voxcpm_lora.py \
        --dataset arbml/arabic_single_speaker_speech_dataset \
        --text-column text \
        --audio-column audio \
        --output-dir ./voxcpm-arabic-lora

The script expects the Hugging Face dataset column schema to contain text and
audio fields. All audio is resampled to 16 kHz mono, encoded with VoxCPM's
AudioVAE to obtain latent patches, and then used to supervise the diffusion
estimator with teacher forcing.

Only the LoRA adapters are saved at the end (``lora_weights.pt`` and
``lora_config.json`` in ``--output-dir``). Load them alongside the base model in
your inference stack and merge or apply at runtime depending on your LoRA
deployment strategy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Audio, load_dataset
from huggingface_hub import login
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from voxcpm import VoxCPM
from voxcpm.model.utils import get_dtype


DEFAULT_DATASET = "arbml/arabic_single_speaker_speech_dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for VoxCPM")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=(
            "Hugging Face dataset identifier (e.g. arbml/arabic_single_speaker_speech_dataset). "
            "Override this if you have validated a different corpus."
        ),
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset configuration name passed to `load_dataset`.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for training.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column name that contains transcription text.",
    )
    parser.add_argument(
        "--audio-column",
        type=str,
        default="audio",
        help="Column name that contains raw audio.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to use (debugging / quick runs).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="openbmb/VoxCPM-0.5B",
        help="Base VoxCPM checkpoint to start from.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory for Hugging Face downloads.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Micro-batch size. Larger values require more GPU memory.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of passes over the dataset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional limit on optimization steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="AdamW learning rate for LoRA parameters.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay applied to LoRA parameters.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold (L2 norm).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often to report training metrics (in steps).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to store the trained LoRA adapters.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Authentication token for private Hugging Face assets. If omitted,"
        " the HF_TOKEN environment variable is used when available.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="Rank of the LoRA matrices (r).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=32.0,
        help="Scaling factor for LoRA updates (alpha).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout applied to the LoRA branch.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=10,
        help="Number of diffusion refinement steps during teacher forcing.",
    )
    parser.add_argument(
        "--stop-loss-weight",
        type=float,
        default=0.1,
        help="Weight applied to the stop-token cross entropy loss.",
    )
    return parser.parse_args()


class LoRALinear(nn.Module):
    """LoRA wrapper around a frozen ``nn.Linear`` layer."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.rank = rank
        self.alpha = alpha
        self.dropout_prob = dropout

        weight = linear.weight.detach().clone()
        bias = linear.bias.detach().clone() if linear.bias is not None else None

        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias, requires_grad=False)

        self.lora_down = nn.Linear(
            linear.in_features,
            rank,
            bias=False,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.lora_up = nn.Linear(
            rank,
            linear.out_features,
            bias=False,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()

        self.scaling = alpha / rank

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original = F.linear(hidden_states, self.weight, self.bias)
        update = self.lora_up(self.lora_down(self.dropout(hidden_states))) * self.scaling
        return original + update


def replace_with_lora(
    root: nn.Module,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    """Replace selected ``nn.Linear`` submodules with ``LoRALinear`` wrappers."""

    target_suffixes = tuple(target_suffixes)
    matched_modules: List[str] = []

    def resolve_parent(module: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
        parts = dotted_name.split(".")
        for attr in parts[:-1]:
            module = getattr(module, attr)
        return module, parts[-1]

    linear_names: List[str] = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(target_suffixes):
            linear_names.append(name)

    for name in linear_names:
        parent, attr = resolve_parent(root, name)
        original: nn.Linear = getattr(parent, attr)
        lora_module = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_module)
        matched_modules.append(name)

    return matched_modules


def save_lora_adapters(model: nn.Module, output_dir: str) -> None:
    """Persist LoRA weights and configuration metadata to ``output_dir``."""

    os.makedirs(output_dir, exist_ok=True)

    state_dict: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, Dict[str, float]] = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state_dict[f"{name}.lora_down.weight"] = module.lora_down.weight.detach().cpu()
            state_dict[f"{name}.lora_up.weight"] = module.lora_up.weight.detach().cpu()
            metadata[name] = {
                "rank": module.rank,
                "alpha": module.alpha,
                "dropout": module.dropout_prob,
            }

    torch.save(state_dict, os.path.join(output_dir, "lora_weights.pt"))
    with open(os.path.join(output_dir, "lora_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


@dataclass
class Batch:
    input_ids: torch.Tensor
    text_mask: torch.Tensor
    latents: torch.Tensor
    latent_mask: torch.Tensor


class VoxCPMArabicDataset(Dataset):
    """Map-style dataset that encodes audio into VoxCPM latent patches."""

    def __init__(
        self,
        dataset,
        model,
        text_column: str,
        audio_column: str,
        target_sr: int,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.text_column = text_column
        self.audio_column = audio_column
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        text = example[self.text_column]
        audio_dict = example[self.audio_column]
        audio = torch.tensor(audio_dict["array"], dtype=torch.float32)
        sr = int(audio_dict["sampling_rate"])

        if audio.ndim > 1:
            audio = audio.mean(dim=0)

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(audio.unsqueeze(0), sr, self.target_sr).squeeze(0)

        audio = audio.unsqueeze(0)
        with torch.no_grad():
            latents = self.model.audio_vae.encode(audio.to(self.model.device), self.target_sr).cpu()

        latents = latents.view(
            self.model.audio_vae.latent_dim,
            -1,
            self.model.patch_size,
        ).permute(1, 2, 0)

        if latents.size(0) > 1:
            latents = latents[:-1]

        token_ids = self.model.text_tokenizer(text)
        token_ids.append(self.model.audio_start_token)
        input_ids = torch.tensor(token_ids, dtype=torch.long)

        return {"input_ids": input_ids, "latents": latents.to(torch.float32)}


def build_dataloader(
    dataset: Dataset,
    model,
    batch_size: int,
) -> DataLoader:
    def collate(samples: List[Dict[str, torch.Tensor]]) -> Batch:
        batch_size = len(samples)
        patch_size = model.patch_size
        latent_dim = model.audio_vae.latent_dim

        max_text = max(sample["input_ids"].size(0) for sample in samples)
        max_patches = max(sample["latents"].size(0) for sample in samples)

        input_ids = torch.full((batch_size, max_text), fill_value=model.audio_end_token, dtype=torch.long)
        text_mask = torch.zeros((batch_size, max_text), dtype=torch.bool)
        latents = torch.zeros((batch_size, max_patches, patch_size, latent_dim), dtype=torch.float32)
        latent_mask = torch.zeros((batch_size, max_patches), dtype=torch.bool)

        for row, sample in enumerate(samples):
            seq_len = sample["input_ids"].size(0)
            patch_len = sample["latents"].size(0)
            input_ids[row, :seq_len] = sample["input_ids"]
            text_mask[row, :seq_len] = True
            latents[row, :patch_len] = sample["latents"]
            latent_mask[row, :patch_len] = True

        return Batch(
            input_ids=input_ids,
            text_mask=text_mask,
            latents=latents,
            latent_mask=latent_mask,
        )

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)


def reset_kv_cache(model) -> None:
    if model.kv_cache is not None:
        model.kv_cache.kv_cache.zero_()
        model.kv_cache.current_length = 0


def teacher_forced_loss(
    model,
    batch: Batch,
    diffusion_steps: int,
    stop_loss_weight: float,
) -> torch.Tensor:
    device = model.device
    input_ids = batch.input_ids.to(device)
    text_mask = batch.text_mask.to(device)
    latents = batch.latents.to(device)
    latent_mask = batch.latent_mask.to(device)

    batch_size = input_ids.size(0)
    max_text = input_ids.size(1)

    feat = torch.zeros(
        batch_size,
        max_text,
        model.patch_size,
        model.audio_vae.latent_dim,
        device=device,
        dtype=model.base_lm.embed_tokens(input_ids[:1]).dtype,
    )

    feat_mask = torch.zeros_like(text_mask, dtype=torch.int32)
    text_mask_int = text_mask.to(torch.int32)

    model.base_lm.train()
    model.residual_lm.train()

    reset_kv_cache(model.base_lm)
    reset_kv_cache(model.residual_lm)

    feat_embed = model.feat_encoder(feat)
    feat_embed = model.enc_to_lm_proj(feat_embed)
    text_embed = model.base_lm.embed_tokens(input_ids)
    combined_embed = text_mask_int.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

    enc_outputs, kv_cache_tuple = model.base_lm(inputs_embeds=combined_embed, is_causal=True)
    model.base_lm.kv_cache.fill_caches(kv_cache_tuple)

    enc_outputs = model.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask_int.unsqueeze(-1)
    lm_hidden = enc_outputs[:, -1, :]

    residual_inputs = enc_outputs + feat_mask.unsqueeze(-1) * feat_embed
    residual_outputs, residual_cache_tuple = model.residual_lm(inputs_embeds=residual_inputs, is_causal=True)
    model.residual_lm.kv_cache.fill_caches(residual_cache_tuple)
    residual_hidden = residual_outputs[:, -1, :]

    prefix_feat_cond = feat[:, -1, ...]

    loss_recon = torch.zeros((), device=device)
    loss_stop = torch.zeros((), device=device)

    for step in range(latents.size(1)):
        active_mask = latent_mask[:, step]
        if not torch.any(active_mask):
            break

        dit_hidden = model.lm_to_dit_proj(lm_hidden) + model.res_to_dit_proj(residual_hidden)
        pred_patch = model.feat_decoder(
            mu=dit_hidden,
            patch_size=model.patch_size,
            cond=prefix_feat_cond.transpose(1, 2).contiguous(),
            n_timesteps=diffusion_steps,
            cfg_value=1.0,
        ).transpose(1, 2)

        target_patch = latents[:, step, :, :]
        loss_recon = loss_recon + F.smooth_l1_loss(
            pred_patch[active_mask].to(torch.float32),
            target_patch[active_mask].to(torch.float32),
        )

        stop_logits = model.stop_head(model.stop_actn(model.stop_proj(lm_hidden)))
        stop_targets = torch.zeros(stop_logits.size(0), dtype=torch.long, device=device)
        if step + 1 < latents.size(1):
            next_active = latent_mask[:, step + 1]
        else:
            next_active = torch.zeros_like(active_mask)
        terminal_mask = active_mask & (~next_active)
        stop_targets[terminal_mask] = 1
        loss_stop = loss_stop + F.cross_entropy(stop_logits, stop_targets)

        teacher_patch = target_patch.detach().to(pred_patch.dtype)
        prefix_feat_cond = teacher_patch

        curr_embed = model.feat_encoder(teacher_patch.unsqueeze(1))
        curr_embed = model.enc_to_lm_proj(curr_embed)

        position_id = torch.tensor([model.base_lm.kv_cache.step()], device=device)
        lm_hidden = model.base_lm.forward_step(curr_embed[:, 0, :], position_id)
        lm_hidden = model.fsq_layer(lm_hidden)

        residual_position_id = torch.tensor([model.residual_lm.kv_cache.step()], device=device)
        residual_hidden = model.residual_lm.forward_step(lm_hidden + curr_embed[:, 0, :], residual_position_id)

    total_loss = loss_recon + stop_loss_weight * loss_stop
    return total_loss


def main() -> None:
    args = parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = VoxCPM.from_pretrained(
        hf_model_id=args.model_id,
        load_denoiser=False,
        cache_dir=args.cache_dir,
    )

    model = pipeline.tts_model
    model.to(device)
    model.audio_vae.to(device)
    model.audio_vae.eval()
    model.audio_vae.requires_grad_(False)
    model.requires_grad_(False)

    lm_dtype = get_dtype(model.config.dtype)
    model.base_lm.setup_cache(args.batch_size, model.config.max_length, model.device, lm_dtype)
    model.residual_lm.setup_cache(args.batch_size, model.config.max_length, model.device, lm_dtype)

    lora_targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "enc_to_lm_proj",
        "lm_to_dit_proj",
        "res_to_dit_proj",
        "stop_proj",
    ]

    matched = replace_with_lora(
        model,
        target_suffixes=lora_targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )

    if not matched:
        raise RuntimeError("No modules matched the requested LoRA targets.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    raw_dataset = load_dataset(
        path=args.dataset,
        name=args.dataset_config,
        split=args.split,
        cache_dir=args.cache_dir,
        use_auth_token=token,
    )

    raw_dataset = raw_dataset.cast_column(args.audio_column, Audio(sampling_rate=16000))
    if args.max_samples:
        raw_dataset = raw_dataset.select(range(args.max_samples))

    dataset = VoxCPMArabicDataset(
        dataset=raw_dataset,
        model=model,
        text_column=args.text_column,
        audio_column=args.audio_column,
        target_sr=16000,
    )

    dataloader = build_dataloader(dataset, model=model, batch_size=args.batch_size)

    global_step = 0

    model.train()

    estimated_total = args.max_steps if args.max_steps else args.num_epochs * max(len(dataloader), 1)
    progress = tqdm(total=estimated_total, desc="training")

    for epoch in range(args.num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            loss = teacher_forced_loss(
                model,
                batch,
                diffusion_steps=args.diffusion_steps,
                stop_loss_weight=args.stop_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()

            global_step += 1
            progress.update(1)
            if global_step % args.log_interval == 0:
                progress.set_postfix({"loss": loss.item()})

            if args.max_steps and global_step >= args.max_steps:
                break

        if args.max_steps and global_step >= args.max_steps:
            break

    progress.close()

    save_lora_adapters(model, args.output_dir)


if __name__ == "__main__":
    main()
