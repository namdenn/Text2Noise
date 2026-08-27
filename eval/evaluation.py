#!/usr/bin/env python3
"""Run speech enhancement with a text-conditioned noise diffusion prior."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import InferenceAlgoRegistry  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate speech enhancement using a speech prior and a "
            "text-conditioned noise diffusion prior."
        )
    )
    parser.add_argument("--segment", type=int, default=-1)
    parser.add_argument("--num_segments", type=int, default=1)
    parser.add_argument("--nbatch", type=int, default=4)
    parser.add_argument("--lambda", dest="lmbd", type=float, default=5.75)
    parser.add_argument("--dataset", choices=("WSJ0", "VB"), default="WSJ0")
    parser.add_argument("--ckpt_path", required=True, help="Speech-prior checkpoint")
    parser.add_argument(
        "--ckpt_noise_path",
        required=True,
        help="Text-conditioned noise checkpoint (v2 for this evaluation)",
    )
    parser.add_argument(
        "--metadata-jsonl",
        "--metadata_jsonl",
        dest="metadata_jsonl",
        required=True,
        help=(
            "Encoded JSONL used to train the noise checkpoint. Stored embeddings "
            "are used directly, as in inference.py."
        ),
    )
    parser.add_argument(
        "--algo_type",
        default="separate_paradiffuseen",
        choices=("separate_paradiffuseen",),
    )
    parser.add_argument("--tag", default="v2_text_noise_prior")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--clean_root", required=True)
    parser.add_argument("--noisy_root", required=True)
    parser.add_argument("--save_root", default="./eval/result")
    parser.add_argument("--num_E", type=int, default=30)
    parser.add_argument("--startstep", type=int, default=0)
    parser.add_argument(
        "--prompt_template",
        default="This is {noise_type} noise",
        help="Must match the caption style used to train v2",
    )
    parser.add_argument("--compute_metrics", action="store_true")
    parser.add_argument("--dnn_mos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.num_segments <= 0:
        parser.error("--num_segments must be positive")
    if args.segment < -1 or args.segment >= args.num_segments:
        parser.error("--segment must be -1 or in [0, num_segments)")
    if args.nbatch <= 0 or args.num_E <= 0:
        parser.error("--nbatch and --num_E must be positive")
    if args.startstep < 0 or args.startstep >= args.num_E:
        parser.error("--startstep must be in [0, num_E)")
    return args


def require_file(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as dataset_file:
        dataset = json.load(dataset_file)
    if not isinstance(dataset, dict):
        raise TypeError("Speech-enhancement evaluation JSON must contain an object")
    return list(dataset.items())


def segment_bounds(num_files, segment, num_segments):
    if segment == -1:
        return 0, num_files
    start = segment * num_files // num_segments
    end = (segment + 1) * num_files // num_segments
    return start, end


def load_stored_embeddings(jsonl_path):
    """Index the exact text conditions stored with the checkpoint's metadata."""
    embeddings = {}
    match_counts = {}

    with open(jsonl_path, "r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            prompt = str(item.get("text", "")).strip()
            if not prompt:
                raise ValueError(
                    f"Metadata row {line_number} in {jsonl_path} has no text prompt"
                )
            if "embedding" not in item:
                raise ValueError(
                    f"Metadata row {line_number} in {jsonl_path} has no embedding"
                )

            embedding = np.asarray(item["embedding"], dtype=np.float32)
            if embedding.shape != (512,):
                raise ValueError(
                    f"Expected a 512-D embedding at row {line_number}, "
                    f"got shape {embedding.shape}"
                )
            if not np.isfinite(embedding).all():
                raise ValueError(
                    f"Stored embedding at row {line_number} contains NaN or infinity"
                )

            if prompt in embeddings and not np.allclose(
                embeddings[prompt], embedding, rtol=0.0, atol=1e-7
            ):
                raise ValueError(
                    f"Prompt {prompt!r} has inconsistent embeddings in {jsonl_path}"
                )
            embeddings[prompt] = embedding
            match_counts[prompt] = match_counts.get(prompt, 0) + 1

    if not embeddings:
        raise ValueError(f"No stored text embeddings were found in {jsonl_path}")

    print(
        f"Loaded {len(embeddings)} stored text conditions from {jsonl_path} "
        f"({sum(match_counts.values())} metadata rows)"
    )
    return embeddings


def get_stored_embedding(embeddings, prompt, nbatch, device, metadata_path):
    """Return one stored condition repeated to match the enhancement batch."""
    normalized_prompt = prompt.strip()
    if normalized_prompt not in embeddings:
        available = ", ".join(sorted(repr(key) for key in embeddings))
        raise ValueError(
            f"Prompt {normalized_prompt!r} was not found in {metadata_path}. "
            "The checkpoint can only use prompts stored in its original encoded "
            f"metadata. Available prompts: {available}"
        )
    embedding = torch.from_numpy(embeddings[normalized_prompt]).to(device)
    return embedding.unsqueeze(0).repeat(nbatch, 1)


def append_rtf(path, row, write_header):
    fieldnames = (
        "speaker_id",
        "file_name",
        "noise_type",
        "snr",
        "duration_seconds",
        "rtf",
    )
    with open(path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_metrics(args, enhanced_dir):
    command = [
        sys.executable,
        "eval/statistics/compute_metrics.py",
        "--enhanced_dir",
        str(enhanced_dir),
        "--data_dir",
        args.data_dir,
        "--save_dir",
        str(enhanced_dir),
        "--dataset",
        args.dataset,
        "--clean_root",
        args.clean_root,
        "--noisy_root",
        args.noisy_root,
    ]
    if args.dnn_mos:
        command.append("--dnn_mos")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def evaluate(args):
    for path, label in (
        (args.data_dir, "evaluation JSON"),
        (args.ckpt_path, "speech checkpoint"),
        (args.ckpt_noise_path, "v2 noise checkpoint"),
        (args.metadata_jsonl, "encoded v2 metadata"),
    ):
        require_file(path, label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.data_dir)
    required_fields = (
        "utt_name",
        "noisy_wav",
        "clean_wav",
        "noise_type",
        "snr",
    )
    for record_key, record in dataset:
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise KeyError(
                f"Evaluation record {record_key!r} is missing fields: {missing}"
            )
    start, end = segment_bounds(len(dataset), args.segment, args.num_segments)
    selected_records = dataset[start:end]
    print(f"Evaluating {len(selected_records)} files at indices [{start}, {end})")

    noise_experiment = Path(args.ckpt_noise_path).parent.name
    save_dir = (
        Path(args.save_root)
        / noise_experiment
        / args.dataset
        / args.algo_type
        / args.tag
    )
    enhanced_dir = save_dir / "speech"
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    stored_embeddings = load_stored_embeddings(args.metadata_jsonl)

    # Fail before loading the large diffusion models if the evaluation asks for
    # a condition that was not used to train this checkpoint.
    evaluation_prompts = {
        args.prompt_template.format(
            noise_type=str(record["noise_type"]).strip().lower()
        ).strip()
        for _, record in dataset
    }
    missing_prompts = sorted(evaluation_prompts - stored_embeddings.keys())
    if missing_prompts:
        raise ValueError(
            f"Evaluation prompts are absent from {args.metadata_jsonl}: "
            f"{missing_prompts}"
        )

    algorithm_class = InferenceAlgoRegistry.get_by_name(args.algo_type)
    enhancer = algorithm_class(
        ckpt_path=args.ckpt_path,
        ckpt_noise=args.ckpt_noise_path,
        num_E=args.num_E,
        verbose=args.verbose,
        device=str(device),
    )

    rtf_name = "rtf.csv" if args.segment == -1 else f"rtf_segment_{args.segment}.csv"
    rtf_path = save_dir / rtf_name

    for _, record in tqdm(selected_records, desc="Speech enhancement"):
        output_path = enhanced_dir / f"{record['utt_name']}.wav"
        if output_path.exists() and not args.overwrite:
            continue

        mixture_path = record["noisy_wav"].format(noisy_root=args.noisy_root)
        clean_path = record["clean_wav"].format(clean_root=args.clean_root)
        require_file(mixture_path, "noisy mixture")
        require_file(clean_path, "clean reference")

        noise_type = str(record["noise_type"]).strip().lower()
        prompt = args.prompt_template.format(noise_type=noise_type)
        text_embedding = get_stored_embedding(
            stored_embeddings,
            prompt,
            args.nbatch,
            device,
            args.metadata_jsonl,
        )

        start_time = time.perf_counter()
        enhanced, _ = enhancer.run(
            mix_file=mixture_path,
            clean_file=None,
            video_file=None,
            text_embedding=text_embedding,
            lmbd=args.lmbd,
            nbatch=args.nbatch,
            startstep=args.startstep,
            wiener_filter=True,
        )
        elapsed = time.perf_counter() - start_time
        enhanced = np.asarray(enhanced)
        if enhanced.ndim != 1:
            raise RuntimeError(
                f"Enhancer returned shape {enhanced.shape} for {record['utt_name']}; "
                "expected a mono waveform"
            )
        if not np.isfinite(enhanced).all():
            raise RuntimeError(
                f"Enhancer returned NaN or infinity for {record['utt_name']}"
            )

        mixture, sample_rate = torchaudio.load(mixture_path)
        if sample_rate != 16000:
            raise ValueError(
                f"Expected 16 kHz audio for {mixture_path}, got {sample_rate} Hz"
            )
        duration = mixture.shape[-1] / sample_rate
        append_rtf(
            rtf_path,
            {
                "speaker_id": record.get("p_id", "unknown"),
                "file_name": record["utt_name"],
                "noise_type": record["noise_type"],
                "snr": record["snr"],
                "duration_seconds": duration,
                "rtf": elapsed / duration,
            },
            write_header=not rtf_path.exists(),
        )
        sf.write(output_path, enhanced, sample_rate)

    expected_outputs = {f"{record['utt_name']}.wav" for _, record in dataset}
    actual_outputs = {path.name for path in enhanced_dir.glob("*.wav")}
    all_complete = expected_outputs.issubset(actual_outputs)
    print(f"Results saved to: {enhanced_dir}")
    print(f"Complete outputs: {len(actual_outputs)}/{len(expected_outputs)}")

    if args.compute_metrics:
        if not all_complete:
            raise RuntimeError(
                "Metrics requested before every segment completed. Run the metric "
                "command after all enhancement jobs have finished."
            )
        run_metrics(args, enhanced_dir)


if __name__ == "__main__":
    evaluate(parse_args())
