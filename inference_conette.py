import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from transformers import RobertaTokenizer

from sgmse.data_module import (
    DEFAULT_TEXT_ENCODER_CHECKPOINT,
    RobertaMLPEncoder,
)
from src import InferenceAlgoRegistry


DEFAULT_DIFFUSION_CHECKPOINT = (
    os.environ.get(
        "DIFFUSION_CHECKPOINT",
        "checkpoints/noise_model.ckpt",
    )
)
DEFAULT_OUTPUT_DIR = (
    os.environ.get("OUTPUT_DIR", "outputs/noise_generation")
)


def encode_prompt(prompt, checkpoint_path, device):
    """Encode a free-form caption exactly as in metadata preprocessing."""
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encoder = RobertaMLPEncoder(checkpoint_path=checkpoint_path).to(device).eval()
    with torch.no_grad():
        tokens = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32,
        ).to(device)
        return encoder(tokens["input_ids"], tokens["attention_mask"])


def load_metadata_condition(jsonl_path, record_index, device):
    """Load an exact training condition for an in-distribution smoke test."""
    if record_index < 0:
        raise ValueError("metadata-index must be non-negative")

    record = None
    current_index = -1
    with open(jsonl_path, "r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            if not line.strip():
                continue
            current_index += 1
            if current_index == record_index:
                record = json.loads(line)
                break

    if record is None:
        raise IndexError(
            f"Metadata record {record_index} was not found in {jsonl_path}"
        )
    if "embedding" not in record:
        raise KeyError(f"Metadata record {record_index} has no 'embedding' field")

    condition = torch.as_tensor(record["embedding"], dtype=torch.float32)
    condition = condition.squeeze()
    if condition.ndim != 1:
        raise ValueError(
            "Stored metadata embedding must reduce to [embedding_dim], "
            f"got {tuple(condition.shape)}"
        )

    caption = next(
        (
            record[key]
            for key in ("text", "caption", "description")
            if key in record and record[key]
        ),
        f"metadata_record_{record_index}",
    )
    return condition.unsqueeze(0).to(device), str(caption)


def plot_and_save_spectrogram(spec, title, save_path):
    magnitude = spec.abs() if torch.is_complex(spec) else spec
    spec_db = 20 * np.log10(magnitude.numpy() + 1e-6)
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def safe_filename(text):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return name[:100] or "conette_noise"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate noise from the model trained on CoNeTTE captions"
    )
    parser.add_argument(
        "--prompt",
        default="A crowd of people are talking in a busy indoor space.",
        help="Use a descriptive CoNeTTE-style caption, not a short class label",
    )
    parser.add_argument(
        "--metadata-jsonl",
        default=None,
        help=(
            "Optional encoded CoNeTTE JSONL. When provided, the stored embedding "
            "is used directly without re-encoding text."
        ),
    )
    parser.add_argument("--metadata-index", type=int, default=0)
    parser.add_argument(
        "--text-checkpoint",
        default=DEFAULT_TEXT_ENCODER_CHECKPOINT,
        help="The exact text checkpoint used to encode the CoNeTTE metadata",
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        default=DEFAULT_DIFFUSION_CHECKPOINT,
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--duration",
        type=float,
        default=2.04,
        help="2.04 seconds reproduces the 256-frame training grid",
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--snr",
        type=float,
        default=0.5,
        help="Target SNR used by the annealed-Langevin corrector",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.snr <= 0:
        parser.error("--snr must be positive")
    if args.duration <= 0:
        parser.error("--duration must be positive")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(args.diffusion_checkpoint):
        raise FileNotFoundError(
            f"Diffusion checkpoint not found: {args.diffusion_checkpoint}"
        )
    if args.metadata_jsonl and not os.path.isfile(args.metadata_jsonl):
        raise FileNotFoundError(
            f"Encoded CoNeTTE metadata not found: {args.metadata_jsonl}"
        )
    if not args.metadata_jsonl and not os.path.isfile(args.text_checkpoint):
        raise FileNotFoundError(
            f"CoNeTTE text encoder checkpoint not found: {args.text_checkpoint}"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if args.metadata_jsonl:
        condition, caption = load_metadata_condition(
            args.metadata_jsonl,
            args.metadata_index,
            device,
        )
        condition_source = (
            f"metadata {args.metadata_jsonl}, record {args.metadata_index}"
        )
    else:
        caption = args.prompt
        condition = encode_prompt(caption, args.text_checkpoint, device)
        condition_source = f"text encoder {args.text_checkpoint}"

    print(f"Caption: {caption}")
    print(f"Condition source: {condition_source}")

    diffuseen_class = InferenceAlgoRegistry.get_by_name("diffuseen")
    engine = diffuseen_class(
        ckpt_path=args.diffusion_checkpoint,
        num_E=args.steps,
        snr=args.snr,
        transform_type="exponent",
        verbose=True,
        device=str(device),
    )
    waveform, generated_spec = engine.prior_sampler(
        condition=condition,
        duration_sec=args.duration,
    )

    if not np.isfinite(waveform).all():
        raise RuntimeError("Generated waveform contains NaN or infinite values")

    output_name = safe_filename(caption)
    audio_path = os.path.join(args.output_dir, f"generated_{output_name}.wav")
    spectrogram_path = os.path.join(
        args.output_dir,
        f"spectrogram_{output_name}.png",
    )
    torchaudio.save(audio_path, torch.from_numpy(waveform).unsqueeze(0), 16000)
    plot_and_save_spectrogram(
        generated_spec,
        f"Generated spectrogram: {caption}",
        spectrogram_path,
    )
    print(f"Audio saved to: {audio_path}")
    print(f"Spectrogram saved to: {spectrogram_path}")


if __name__ == "__main__":
    main()
