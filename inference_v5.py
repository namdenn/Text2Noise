#!/usr/bin/env python3
"""Generate noise with CLAP text and an original OUVE score checkpoint."""

import argparse
import json
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from sgmse.clap_conditioning import DEFAULT_CLAP_MODEL, FrozenCLAPConditioner
from src import InferenceAlgoRegistry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate noise using CLAP text conditioning and OUVE diffusion"
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--checkpoint", "--diffusion-checkpoint", dest="checkpoint", required=True
    )
    parser.add_argument("--output-dir", default="outputs/clap_audio_score_v5")
    parser.add_argument(
        "--conditioning-config",
        default=None,
        help="conditioning_config.json written during audio metadata preparation",
    )
    parser.add_argument("--clap-model", default=None)
    parser.add_argument("--clap-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--snr", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_clap_config(args):
    if not args.conditioning_config:
        warnings.warn(
            "No --conditioning-config supplied; using the requested/default CLAP "
            "checkpoint. Pass the training metadata's conditioning_config.json "
            "to guarantee that the audio and text towers match."
        )
        return args.clap_model or DEFAULT_CLAP_MODEL, args.clap_revision, 512

    with open(args.conditioning_config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("conditioning") != "clap_audio":
        raise ValueError(
            f"{args.conditioning_config} is not a CLAP-audio conditioning config"
        )
    stored_model = config.get("model_name_or_path")
    stored_revision = config.get("revision")
    resolved_revision = config.get("resolved_revision") or stored_revision
    if not stored_model:
        raise ValueError("Conditioning config has no model_name_or_path")
    if args.clap_model and args.clap_model != stored_model:
        raise ValueError(
            f"--clap-model {args.clap_model!r} differs from training model "
            f"{stored_model!r}"
        )
    if args.clap_revision and args.clap_revision != stored_revision:
        raise ValueError(
            f"--clap-revision {args.clap_revision!r} differs from training "
            f"revision {stored_revision!r}"
        )
    return stored_model, resolved_revision, int(config.get("embedding_dim", 512))


def save_spectrogram(spec, path, title):
    magnitude = spec.detach().abs().float().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(
        20 * np.log10(magnitude + 1e-6),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time frames")
    plt.ylabel("Frequency bins")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    if args.duration <= 0 or args.steps <= 0:
        raise ValueError("--duration and --steps must be positive")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    clap_model, clap_revision, embedding_dim = resolve_clap_config(args)

    # Encode first, then release CLAP so diffusion gets the full GPU budget.
    clap = FrozenCLAPConditioner(
        model_name_or_path=clap_model,
        revision=clap_revision,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    condition = clap.encode_text(args.prompt).cpu()
    if condition.shape != (1, embedding_dim):
        raise RuntimeError(
            f"Expected one {embedding_dim}-D CLAP text embedding, "
            f"got {tuple(condition.shape)}"
        )
    del clap
    if device.type == "cuda":
        torch.cuda.empty_cache()

    diffuse_cls = InferenceAlgoRegistry.get_by_name("diffuseen")
    engine = diffuse_cls(
        ckpt_path=args.checkpoint,
        num_E=args.steps,
        transform_type="exponent",
        eps=args.eps,
        snr=args.snr,
        sr=args.sample_rate,
        verbose=True,
        device=str(device),
    )
    sde_name = getattr(engine.model.sde, "__name__", "")
    if sde_name != "ouve":
        raise ValueError(
            "This entrypoint requires an original OUVE score-model checkpoint; "
            f"got sde={sde_name or type(engine.model.sde).__name__}"
        )

    # Seed after constructing both pretrained models so the generation seed is
    # independent of any initialization performed by their loaders.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    waveform, transformed_spec = engine.prior_sampler(
        condition=condition.to(device),
        duration_sec=args.duration,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", args.prompt).strip("_") or "prompt"
    audio_path = os.path.join(args.output_dir, f"generated_{slug}.wav")
    image_path = os.path.join(args.output_dir, f"spectrogram_{slug}.png")
    audio_tensor = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
    torchaudio.save(audio_path, audio_tensor, args.sample_rate)
    save_spectrogram(
        transformed_spec,
        image_path,
        f"Generated noise: {args.prompt}",
    )
    print(f"CLAP text embedding norm: {condition.norm(dim=-1).item():.6f}")
    print(f"Saved audio to {audio_path}")
    print(f"Saved spectrogram to {image_path}")


if __name__ == "__main__":
    main()
