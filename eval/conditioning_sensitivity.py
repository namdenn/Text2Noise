#!/usr/bin/env python3
"""Probe whether prior noise samples respond to label conditioning.

For each random seed, this script generates one prior sample per label while
resetting the RNG before every label. If conditioning matters, samples generated
from the same seed should move when the label embedding changes.
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import RobertaModel, RobertaTokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import InferenceAlgoRegistry  # noqa: E402


class MLPLayers(nn.Module):
    def __init__(self, units=(768, 512, 512), nonlin=nn.ReLU(), dropout=0.1):
        super().__init__()
        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(nonlin)
            sequence.append(nn.Dropout(dropout))
        self.sequential = nn.Sequential(*sequence[:-2])

    def forward(self, x):
        return self.sequential(x)


class RobertaMLPEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, model_name_or_path="roberta-base", local_files_only=False):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self.mlp = MLPLayers()
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.mlp(outputs.pooler_output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_diffusion", type=str, required=True)
    parser.add_argument("--ckpt_text", type=str, required=True)
    parser.add_argument("--text_model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="eval/result/conditioning_sensitivity")
    parser.add_argument("--labels", nargs="+", default=["babble", "car", "cafe", "street", "lr", "white"])
    parser.add_argument("--prompt_template", type=str, default="This is {label} noise")
    parser.add_argument("--num_seeds", type=int, default=4)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--duration_sec", type=float, default=4.0)
    parser.add_argument("--num_E", type=int, default=50)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--transform_type", type=str, default="exponent")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--include_null", action="store_true")
    parser.add_argument("--include_random", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--random_embedding_seed", type=int, default=1234)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slugify(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "condition"


def encode_prompt(tokenizer, text_model, prompt, device):
    with torch.no_grad():
        tok = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        ).to(device)
        emb = text_model(tok["input_ids"], tok["attention_mask"])
    return emb


def waveform_features(wav, sample_rate):
    wav = wav.detach().float().cpu().reshape(-1)
    wav = wav - wav.mean()
    rms = torch.sqrt(torch.mean(wav.pow(2)) + 1e-12).item()
    zcr = (wav[:-1] * wav[1:] < 0).float().mean().item() if wav.numel() > 1 else 0.0

    n_fft = 1024
    hop_length = 256
    window = torch.hann_window(n_fft)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    mag = spec.abs() + 1e-8
    logmag = torch.log1p(mag)
    freqs = torch.linspace(0, sample_rate / 2, mag.shape[0])
    frame_energy = mag.sum(dim=0) + 1e-8
    centroid = ((freqs[:, None] * mag).sum(dim=0) / frame_energy).mean().item()
    flatness = (torch.exp(torch.log(mag).mean(dim=0)) / mag.mean(dim=0)).mean().item()

    return {
        "rms": rms,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "spectral_flatness": flatness,
        "logmag": logmag,
    }


def spectral_distance(a, b):
    x = a["logmag"].reshape(-1)
    y = b["logmag"].reshape(-1)
    n = min(x.numel(), y.numel())
    x = x[:n]
    y = y[:n]
    l1 = torch.mean(torch.abs(x - y)).item()
    l2 = torch.sqrt(torch.mean((x - y).pow(2)) + 1e-12).item()
    cosine = torch.nn.functional.cosine_similarity(x[None, :], y[None, :]).item()
    return {
        "logmag_l1": l1,
        "logmag_l2": l2,
        "logmag_cosine_distance": 1.0 - cosine,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_distances(rows):
    summary = {}
    for key in ("logmag_l1", "logmag_l2", "logmag_cosine_distance"):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean()) if values.size else None
        summary[f"{key}_std"] = float(values.std()) if values.size else None
    return summary


def main():
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    wav_dir = os.path.join(args.output_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    try:
        print(f"Loading text encoder from: {args.text_model_name_or_path}", flush=True)
        tokenizer = RobertaTokenizer.from_pretrained(
            args.text_model_name_or_path,
            local_files_only=args.local_files_only,
        )
        text_model = RobertaMLPEncoder(
            checkpoint_path=args.ckpt_text,
            model_name_or_path=args.text_model_name_or_path,
            local_files_only=args.local_files_only,
        ).to(device).eval()
    except OSError as exc:
        mode = "offline cache/local path" if args.local_files_only else "Hugging Face or local path"
        raise SystemExit(
            f"Could not load RoBERTa resources from {args.text_model_name_or_path!r} "
            f"using {mode}. If this machine is offline, set --text_model_name_or_path "
            "to a local roberta-base directory or pre-populate the Hugging Face cache. "
            "If network access is available, run without --local_files_only."
        ) from exc

    conditions = []
    for label in args.labels:
        prompt = args.prompt_template.format(label=label)
        conditions.append({
            "name": label,
            "kind": "label",
            "prompt": prompt,
            "embedding": encode_prompt(tokenizer, text_model, prompt, device),
        })

    if args.include_null:
        conditions.append({
            "name": "null",
            "kind": "null",
            "prompt": "",
            "embedding": torch.zeros_like(conditions[0]["embedding"]),
        })

    if args.include_random:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.random_embedding_seed)
        random_emb = torch.randn(conditions[0]["embedding"].shape, generator=gen).to(device)
        conditions.append({
            "name": "random",
            "kind": "random",
            "prompt": "",
            "embedding": random_emb,
        })

    print(f"Loading diffusion checkpoint from: {args.ckpt_diffusion}", flush=True)
    diffuse_cls = InferenceAlgoRegistry.get_by_name("diffuseen")
    engine = diffuse_cls(
        ckpt_path=args.ckpt_diffusion,
        num_E=args.num_E,
        transform_type=args.transform_type,
        sr=args.sample_rate,
        verbose=args.verbose,
        device=device,
    )

    manifest_rows = []
    generated = {}
    seeds = [args.seed_start + i for i in range(args.num_seeds)]

    for seed in seeds:
        generated[seed] = {}
        for condition in conditions:
            if args.verbose:
                print(f"Generating seed={seed} condition={condition['name']}", flush=True)
            set_seed(seed)
            wav, _ = engine.prior_sampler(
                condition=condition["embedding"],
                duration_sec=args.duration_sec,
            )
            wav_tensor = torch.as_tensor(wav, dtype=torch.float32).reshape(1, -1)
            file_name = f"seed_{seed:04d}_{slugify(condition['name'])}.wav"
            wav_path = os.path.join(wav_dir, file_name)
            torchaudio.save(wav_path, wav_tensor.cpu(), args.sample_rate)

            feats = waveform_features(wav_tensor, args.sample_rate)
            generated[seed][condition["name"]] = {
                "condition": condition,
                "wav_path": wav_path,
                "features": feats,
            }
            manifest_rows.append({
                "seed": seed,
                "condition": condition["name"],
                "kind": condition["kind"],
                "prompt": condition["prompt"],
                "wav_path": wav_path,
                "rms": feats["rms"],
                "zero_crossing_rate": feats["zero_crossing_rate"],
                "spectral_centroid_hz": feats["spectral_centroid_hz"],
                "spectral_flatness": feats["spectral_flatness"],
            })

    pairwise_rows = []
    for seed in seeds:
        for name_a, name_b in combinations(generated[seed].keys(), 2):
            dist = spectral_distance(generated[seed][name_a]["features"], generated[seed][name_b]["features"])
            pairwise_rows.append({
                "comparison": "same_seed_different_condition",
                "seed_a": seed,
                "seed_b": seed,
                "condition_a": name_a,
                "condition_b": name_b,
                **dist,
            })

    within_condition_rows = []
    for condition in conditions:
        name = condition["name"]
        for seed_a, seed_b in combinations(seeds, 2):
            dist = spectral_distance(generated[seed_a][name]["features"], generated[seed_b][name]["features"])
            within_condition_rows.append({
                "comparison": "same_condition_different_seed",
                "seed_a": seed_a,
                "seed_b": seed_b,
                "condition_a": name,
                "condition_b": name,
                **dist,
            })

    all_distance_rows = pairwise_rows + within_condition_rows
    write_csv(
        os.path.join(args.output_dir, "manifest.csv"),
        manifest_rows,
        [
            "seed",
            "condition",
            "kind",
            "prompt",
            "wav_path",
            "rms",
            "zero_crossing_rate",
            "spectral_centroid_hz",
            "spectral_flatness",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "pairwise_distances.csv"),
        all_distance_rows,
        [
            "comparison",
            "seed_a",
            "seed_b",
            "condition_a",
            "condition_b",
            "logmag_l1",
            "logmag_l2",
            "logmag_cosine_distance",
        ],
    )

    summary = {
        "args": vars(args),
        "num_generated": len(manifest_rows),
        "same_seed_different_condition": summarize_distances(pairwise_rows),
        "same_condition_different_seed": summarize_distances(within_condition_rows),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved generated WAVs and metrics to: {args.output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()