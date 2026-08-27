#!/usr/bin/env python3
"""Prepare v5 JSONL metadata with frozen CLAP audio embeddings."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from sgmse.clap_conditioning import DEFAULT_CLAP_MODEL, FrozenCLAPConditioner


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Encode each metadata row's noise waveform with CLAP's audio tower. "
            "The output keeps the schema expected by SpecsDataModule."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--splits", nargs="+", default=("train", "val", "test")
    )
    parser.add_argument("--model-name-or-path", default=DEFAULT_CLAP_MODEL)
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face commit/tag. Use the same value at inference.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-audio-seconds", type=float, default=10.0)
    parser.add_argument("--expected-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_jsonl(path):
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                item = json.loads(value)
            except json.JSONDecodeError:
                # Also accept manifests containing one unquoted audio path per
                # line, even though strict JSONL normally contains objects.
                item = {"wav_path": value}
            if isinstance(item, str):
                item = {"wav_path": item}
            if not isinstance(item, dict):
                raise TypeError(
                    f"{path}:{line_number} must be a JSON object or audio path, "
                    f"got {type(item).__name__}"
                )
            wav_path = str(item.get("wav_path", "")).strip()
            if not wav_path:
                raise ValueError(f"{path}:{line_number} has no wav_path")
            resolved_wav_path = Path(wav_path).expanduser()
            if not resolved_wav_path.is_absolute():
                resolved_wav_path = path.parent / resolved_wav_path
            resolved_wav_path = resolved_wav_path.resolve()
            if not resolved_wav_path.is_file():
                raise FileNotFoundError(
                    f"Audio referenced by {path}:{line_number} does not exist: "
                    f"{resolved_wav_path}"
                )
            item["wav_path"] = str(resolved_wav_path)
            items.append(item)
    if not items:
        raise ValueError(f"No metadata rows found in {path}")
    return items


def load_mono_resampled(path, target_sample_rate):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio file: {path}")
    waveform = waveform.float().mean(dim=0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, target_sample_rate
        )
    if not torch.isfinite(waveform).all():
        raise ValueError(f"Audio contains NaN or infinity: {path}")
    return waveform.numpy()


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def requested_config(args):
    return {
        "conditioning": "clap_audio",
        "model_name_or_path": args.model_name_or_path,
        "revision": args.revision,
        "embedding_dim": args.expected_dim,
        "normalized": True,
        "max_audio_seconds": args.max_audio_seconds,
        "seed": args.seed,
    }


def handle_existing_outputs(args, output_dir):
    output_paths = [output_dir / f"{split}.jsonl" for split in args.splits]
    existing = [path for path in output_paths if path.exists()]
    if not existing or args.overwrite:
        return False
    if not args.skip_existing:
        raise FileExistsError(
            f"Output exists: {existing[0]}. Pass --overwrite or --skip-existing."
        )
    if len(existing) != len(output_paths):
        raise RuntimeError(
            "Only some CLAP metadata splits exist. Use --overwrite to rebuild "
            "all splits and avoid mixing conditioning configurations."
        )

    config_path = output_dir / "conditioning_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Existing metadata has no {config_path.name}; use --overwrite to rebuild it."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        existing_config = json.load(handle)
    mismatches = {
        key: (existing_config.get(key), value)
        for key, value in requested_config(args).items()
        if existing_config.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: stored={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(
            f"Existing CLAP metadata configuration differs ({details}). "
            "Use --overwrite to re-encode it."
        )
    print(f"All requested CLAP metadata already exists in {output_dir}")
    return True


def encode_split(args, conditioner, input_path, output_path):
    if output_path.exists():
        if args.skip_existing:
            print(f"Skipping existing {output_path}")
            return
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {output_path}. Pass --overwrite or --skip-existing."
            )

    items = read_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    max_length = round(args.max_audio_seconds * conditioner.sample_rate)

    try:
        with temp_path.open("w", encoding="utf-8") as output_handle:
            progress = tqdm(
                batched(items, args.batch_size),
                total=(len(items) + args.batch_size - 1) // args.batch_size,
                desc=input_path.stem,
            )
            for batch in progress:
                waveforms = [
                    load_mono_resampled(item["wav_path"], conditioner.sample_rate)
                    for item in batch
                ]
                embeddings = conditioner.encode_audio(
                    waveforms, max_length=max_length
                ).cpu()
                if embeddings.shape != (len(batch), args.expected_dim):
                    raise RuntimeError(
                        "Unexpected CLAP embedding shape: "
                        f"expected {(len(batch), args.expected_dim)}, "
                        f"got {tuple(embeddings.shape)}"
                    )
                for item, embedding in zip(batch, embeddings):
                    item["embedding"] = embedding.tolist()
                    item["embedding_source"] = "clap_audio"
                    output_handle.write(json.dumps(item) + "\n")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_audio_seconds <= 0:
        raise ValueError("--max-audio-seconds must be positive")

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")
    if handle_existing_outputs(args, output_dir):
        return
    for split in args.splits:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(f"Required input metadata not found: {input_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    conditioner = FrozenCLAPConditioner(
        model_name_or_path=args.model_name_or_path,
        revision=args.revision,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    print(
        f"Loaded CLAP {args.model_name_or_path!r} on {conditioner.device}; "
        f"audio sample rate={conditioner.sample_rate} Hz, "
        f"truncation={conditioner.audio_truncation}, "
        f"padding={conditioner.audio_padding}"
    )

    for split in args.splits:
        encode_split(
            args,
            conditioner,
            input_dir / f"{split}.jsonl",
            output_dir / f"{split}.jsonl",
        )

    config = {
        **requested_config(args),
        "resolved_revision": conditioner.resolved_revision,
        "audio_sample_rate": conditioner.sample_rate,
        "audio_truncation": conditioner.audio_truncation,
        "audio_padding": conditioner.audio_padding,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "conditioning_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print(f"Wrote CLAP-audio metadata to {output_dir}")


if __name__ == "__main__":
    main()
