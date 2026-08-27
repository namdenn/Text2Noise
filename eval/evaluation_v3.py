#!/usr/bin/env python3
"""Run speech enhancement with stored v3 Conette text conditions."""

import argparse
import csv
import hashlib
import json
import os
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

from eval.evaluation import (  # noqa: E402
    append_rtf,
    load_dataset,
    require_file,
    run_metrics,
    segment_bounds,
)
from src import InferenceAlgoRegistry  # noqa: E402


NOISE_PATH_FIELDS = (
    "noise_wav",
    "noise_path",
    "noise_audio_path",
    "reference_noise_wav",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate speech enhancement with the v3 noise prior using stored "
            "Conette-caption embeddings."
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
        help="v3 text-conditioned noise checkpoint",
    )
    parser.add_argument(
        "--metadata-jsonl",
        "--metadata_jsonl",
        dest="metadata_jsonl",
        required=True,
        help="Encoded v3 JSONL containing Conette captions and stored embeddings",
    )
    parser.add_argument(
        "--condition-selection",
        choices=("auto", "exact", "class-deterministic"),
        default="exact",
        help=(
            "exact matches the source path or the utterance/noise/speaker identity; "
            "auto falls back to a reproducible stored caption from the same noise "
            "class; class-deterministic always uses that fallback"
        ),
    )
    parser.add_argument(
        "--algo_type",
        default="separate_paradiffuseen",
        choices=("separate_paradiffuseen",),
    )
    parser.add_argument("--tag", default="v3_conette_stored_embeddings")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--clean_root", required=True)
    parser.add_argument("--noisy_root", required=True)
    parser.add_argument("--save_root", default="./eval/result")
    parser.add_argument("--num_E", type=int, default=30)
    parser.add_argument("--startstep", type=int, default=0)
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


def normalize_noise_type(value):
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "livingroom": "lr",
        "living_room": "lr",
    }
    return aliases.get(normalized, normalized)


def normalize_audio_path(value):
    return os.path.normcase(os.path.normpath(os.path.abspath(str(value))))


def get_caption(record, line_number, metadata_path):
    for key in ("text", "caption", "description"):
        if record.get(key):
            return str(record[key]).strip()
    raise ValueError(
        f"Metadata row {line_number} in {metadata_path} has no Conette caption"
    )


def infer_metadata_noise_type(record, known_noise_types):
    for key in ("noise_type", "category", "class", "label"):
        if record.get(key):
            candidate = normalize_noise_type(record[key])
            if candidate in known_noise_types:
                return candidate

    path_tokens = {
        normalize_noise_type(token)
        for token in str(record["wav_path"]).replace("\\", "/").split("/")
        if token
    }
    matches = sorted(path_tokens & known_noise_types)
    if len(matches) == 1:
        return matches[0]
    return None


def metadata_identity(wav_path, noise_type):
    """Build the identity shared by v3 metadata and ntcd_timit.json."""
    path_parts = [
        token
        for token in str(wav_path).replace("\\", "/").split("/")
        if token
    ]
    if len(path_parts) < 2:
        return None
    utterance_stem = os.path.splitext(path_parts[-1])[0].strip().lower()
    speaker_id = path_parts[-2].strip().lower()
    if not utterance_stem or not speaker_id or noise_type is None:
        return None
    return utterance_stem, normalize_noise_type(noise_type), speaker_id


def evaluation_identity(record):
    """Parse e.g. sa1_Babble_09F into (sa1, babble, 09f)."""
    noise_type = normalize_noise_type(record["noise_type"])
    speaker_id = str(record["p_id"]).strip().lower()
    utterance_name = str(record["utt_name"]).strip()
    suffix = f"_{noise_type}_{speaker_id}"
    if not utterance_name.lower().endswith(suffix):
        raise ValueError(
            f"Cannot derive a v3 metadata identity from utt_name "
            f"{utterance_name!r}, noise_type {record['noise_type']!r}, and "
            f"p_id {record['p_id']!r}"
        )
    utterance_stem = utterance_name[: -len(suffix)].strip().lower()
    if not utterance_stem:
        raise ValueError(f"Empty utterance stem in evaluation record {utterance_name!r}")
    return utterance_stem, noise_type, speaker_id


def load_conette_catalog(metadata_path, noise_types):
    """Load v3 rows and index their stored conditions by path and noise class."""
    known_noise_types = {normalize_noise_type(value) for value in noise_types}
    by_path = {}
    by_basename = {}
    by_identity = {}
    by_noise_type = {noise_type: [] for noise_type in known_noise_types}
    unclassified = 0

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata_index = -1
        for line_number, line in enumerate(metadata_file, start=1):
            if not line.strip():
                continue
            metadata_index += 1
            record = json.loads(line)
            missing = [key for key in ("wav_path", "embedding") if key not in record]
            if missing:
                raise KeyError(
                    f"Metadata row {line_number} in {metadata_path} is missing "
                    f"fields: {missing}"
                )

            embedding = np.asarray(record["embedding"], dtype=np.float32).squeeze()
            if embedding.shape != (512,):
                raise ValueError(
                    f"Expected a 512-D embedding at metadata row {line_number}, "
                    f"got {embedding.shape}"
                )
            if not np.isfinite(embedding).all():
                raise ValueError(
                    f"Embedding at metadata row {line_number} contains NaN or infinity"
                )

            condition = {
                "metadata_index": metadata_index,
                "wav_path": str(record["wav_path"]),
                "caption": get_caption(record, line_number, metadata_path),
                "embedding": embedding,
            }
            normalized_path = normalize_audio_path(condition["wav_path"])
            if normalized_path in by_path:
                raise ValueError(
                    f"Duplicate v3 metadata audio path: {condition['wav_path']}"
                )
            by_path[normalized_path] = condition
            basename = os.path.basename(normalized_path)
            by_basename.setdefault(basename, []).append(condition)

            noise_type = infer_metadata_noise_type(record, known_noise_types)
            condition["noise_type"] = noise_type
            if noise_type is None:
                unclassified += 1
            else:
                by_noise_type[noise_type].append(condition)
                identity = metadata_identity(condition["wav_path"], noise_type)
                if identity in by_identity:
                    raise ValueError(
                        "Duplicate v3 metadata identity "
                        f"{identity}: {condition['wav_path']}"
                    )
                by_identity[identity] = condition

    if not by_path:
        raise ValueError(f"No stored v3 conditions were found in {metadata_path}")

    missing_classes = sorted(
        noise_type
        for noise_type, conditions in by_noise_type.items()
        if not conditions
    )
    if missing_classes:
        raise ValueError(
            "Could not find v3 metadata rows for evaluation noise classes: "
            f"{missing_classes}. The class must appear in a metadata field or "
            "as a directory in wav_path."
        )

    for conditions in by_noise_type.values():
        conditions.sort(
            key=lambda condition: (
                normalize_audio_path(condition["wav_path"]),
                condition["metadata_index"],
            )
        )

    counts = ", ".join(
        f"{noise_type}={len(by_noise_type[noise_type])}"
        for noise_type in sorted(by_noise_type)
    )
    print(
        f"Loaded {len(by_path)} stored v3 Conette conditions from {metadata_path} "
        f"({counts}; exact identities={len(by_identity)}; "
        f"unclassified={unclassified})"
    )
    return {
        "by_path": by_path,
        "by_basename": by_basename,
        "by_identity": by_identity,
        "by_noise_type": by_noise_type,
    }


def find_exact_condition(catalog, record):
    for field in NOISE_PATH_FIELDS:
        source_path = record.get(field)
        if not source_path:
            continue
        normalized_path = normalize_audio_path(source_path)
        condition = catalog["by_path"].get(normalized_path)
        if condition is not None:
            return condition, f"exact:{field}"

        basename_matches = catalog["by_basename"].get(
            os.path.basename(normalized_path), []
        )
        if len(basename_matches) == 1:
            return basename_matches[0], f"exact-basename:{field}"
    return None, None


def select_condition(catalog, record_key, record, selection, metadata_path):
    if selection in ("auto", "exact"):
        condition, source = find_exact_condition(catalog, record)
        if condition is not None:
            return condition, source

        identity = evaluation_identity(record)
        condition = catalog["by_identity"].get(identity)
        if condition is not None:
            return condition, "exact:utterance-noise-speaker"

        if selection == "exact":
            raise ValueError(
                f"Evaluation record {record_key!r} has no exact stored condition "
                f"in {metadata_path}. Expected identity {identity}, or one of the "
                f"source-path fields {NOISE_PATH_FIELDS}."
            )

    noise_type = normalize_noise_type(record["noise_type"])
    candidates = catalog["by_noise_type"].get(noise_type, [])
    if not candidates:
        raise ValueError(
            f"No stored v3 Conette conditions are available for {noise_type!r}"
        )
    digest = hashlib.sha256(str(record_key).encode("utf-8")).digest()
    condition_index = int.from_bytes(digest[:8], "big") % len(candidates)
    return candidates[condition_index], "class-deterministic"


def validate_eval_record(record_key, record):
    required_fields = (
        "utt_name",
        "noisy_wav",
        "clean_wav",
        "noise_type",
        "p_id",
        "snr",
    )
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise KeyError(
            f"Evaluation record {record_key!r} is missing fields: {missing}"
        )


def append_condition_audit(path, record, condition, selection, write_header):
    fieldnames = (
        "utt_name",
        "noise_type",
        "selection",
        "metadata_index",
        "metadata_wav_path",
        "conette_caption",
    )
    with open(path, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "utt_name": record["utt_name"],
                "noise_type": record["noise_type"],
                "selection": selection,
                "metadata_index": condition["metadata_index"],
                "metadata_wav_path": condition["wav_path"],
                "conette_caption": condition["caption"],
            }
        )


def evaluate(args):
    for path, label in (
        (args.data_dir, "evaluation JSON"),
        (args.ckpt_path, "speech checkpoint"),
        (args.ckpt_noise_path, "v3 noise checkpoint"),
        (args.metadata_jsonl, "encoded v3 Conette metadata"),
    ):
        require_file(path, label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(args.data_dir)
    for record_key, record in dataset:
        validate_eval_record(record_key, record)
    start, end = segment_bounds(len(dataset), args.segment, args.num_segments)
    selected_records = dataset[start:end]
    print(f"Evaluating {len(selected_records)} files at indices [{start}, {end})")

    catalog = load_conette_catalog(
        args.metadata_jsonl,
        {record["noise_type"] for _, record in dataset},
    )
    selected_conditions = {
        record_key: select_condition(
            catalog,
            record_key,
            record,
            args.condition_selection,
            args.metadata_jsonl,
        )
        for record_key, record in selected_records
    }

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

    algorithm_class = InferenceAlgoRegistry.get_by_name(args.algo_type)
    enhancer = algorithm_class(
        ckpt_path=args.ckpt_path,
        ckpt_noise=args.ckpt_noise_path,
        num_E=args.num_E,
        verbose=args.verbose,
        device=str(device),
    )

    suffix = "" if args.segment == -1 else f"_segment_{args.segment}"
    rtf_path = save_dir / f"rtf{suffix}.csv"
    condition_audit_path = save_dir / f"conditions{suffix}.csv"

    for record_key, record in tqdm(selected_records, desc="v3 speech enhancement"):
        output_path = enhanced_dir / f"{record['utt_name']}.wav"
        if output_path.exists() and not args.overwrite:
            continue

        mixture_path = record["noisy_wav"].format(noisy_root=args.noisy_root)
        clean_path = record["clean_wav"].format(clean_root=args.clean_root)
        require_file(mixture_path, "noisy mixture")
        require_file(clean_path, "clean reference")

        condition, selection = selected_conditions[record_key]
        text_embedding = (
            torch.from_numpy(condition["embedding"])
            .to(device)
            .unsqueeze(0)
            .repeat(args.nbatch, 1)
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
        append_condition_audit(
            condition_audit_path,
            record,
            condition,
            selection,
            write_header=not condition_audit_path.exists(),
        )

    expected_outputs = {f"{record['utt_name']}.wav" for _, record in dataset}
    actual_outputs = {path.name for path in enhanced_dir.glob("*.wav")}
    all_complete = expected_outputs.issubset(actual_outputs)
    print(f"Results saved to: {enhanced_dir}")
    print(f"Condition audit saved to: {condition_audit_path}")
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
