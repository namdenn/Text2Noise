import os
import argparse
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
from src import InferenceAlgoRegistry

# def convert_spec_to_audio(spec_tensor, length, n_fft=512, hop_length=128):
#     """
#     Decompresses complex spectrogram features using proper torch.sgn
#     and reconstructs the time-domain waveform using iSTFT.
#     """
#     scaled_spec = torch.sgn(spec_tensor) * (torch.abs(spec_tensor) ** (1 / 0.5))
    
#     window = torch.hann_window(n_fft).to(scaled_spec.device)
#     waveform = torch.istft(
#         scaled_spec,
#         n_fft=n_fft,
#         hop_length=hop_length,
#         window=window,
#         length=length
#     )
#     return waveform.cpu().numpy().reshape(1, -1)

def plot_and_save_spectrogram(spec_data, title="Spectrogram", save_path="outputs/spectrogram.png"):
    if torch.is_complex(spec_data):
        spec_data = torch.abs(spec_data)
    spec_np = spec_data.numpy()
    spec_db = 20 * np.log10(spec_np + 1e-6)

    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Spectrogram image saved to: {save_path}")
    plt.close()


def load_stored_embedding(jsonl_path, prompt, device):
    """Load the exact conditioning vector used to train an existing checkpoint."""
    reference = None
    matches = 0

    with open(jsonl_path, "r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            if item.get("text", "").strip() != prompt.strip():
                continue

            if "embedding" not in item:
                raise ValueError(
                    f"Matching row {line_number} in {jsonl_path} has no embedding"
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

            if reference is None:
                reference = embedding
            elif not np.allclose(reference, embedding, rtol=0.0, atol=1e-7):
                raise ValueError(
                    f"Prompt {prompt!r} has inconsistent embeddings in {jsonl_path}"
                )
            matches += 1

    if reference is None:
        raise ValueError(
            f"Prompt {prompt!r} was not found in {jsonl_path}. The existing "
            "checkpoints can only use prompts and embeddings stored in their "
            "original encoded metadata."
        )

    print(
        f"Loaded stored condition for {prompt!r}: matches={matches}, "
        f"shape={reference.shape}, mean={reference.mean():.6f}, "
        f"std={reference.std():.6f}, norm={np.linalg.norm(reference):.6f}"
    )
    return torch.from_numpy(reference).unsqueeze(0).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a noise waveform from text")
    parser.add_argument("--prompt", default="This is babble noise")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument(
        "--metadata-jsonl",
        required=True,
        help="Encoded JSONL used to train the selected diffusion checkpoint",
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        default=os.environ.get(
            "DIFFUSION_CHECKPOINT",
            "checkpoints/noise_model.ckpt",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OUTPUT_DIR", "outputs/noise_generation"),
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--snr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    prompt_str = args.prompt
    prompt_tensor = load_stored_embedding(
        jsonl_path=args.metadata_jsonl,
        prompt=prompt_str,
        device=device,
    )

    DiffUSEEN = InferenceAlgoRegistry.get_by_name("diffuseen")
    engine = DiffUSEEN(
        ckpt_path=args.diffusion_checkpoint,
        num_E=args.steps,
        transform_type="exponent",
        snr=args.snr,
        verbose=True,
        device=str(device),
    )

    # Seed after model construction so every prompt uses the same prior and
    # corrector noise when the command is repeated with the same seed.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    audio_waveform, raw_spec = engine.prior_sampler(
        condition=prompt_tensor,
        duration_sec=args.duration,
    )

    file_safe_prompt = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt_str).strip("_")
    output_audio_path = os.path.join(args.output_dir, f"generated_{file_safe_prompt}.wav")
    output_spec_path = os.path.join(args.output_dir, f"spectrogram_{file_safe_prompt}.png")

    torchaudio.save(output_audio_path, torch.tensor(audio_waveform).unsqueeze(0), 16000)
    plot_and_save_spectrogram(
        spec_data=raw_spec, 
        title=f"Generated Spectrogram: '{prompt_str}'",
        save_path=output_spec_path
    )
