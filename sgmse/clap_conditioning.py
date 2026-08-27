"""Shared CLAP audio/text conditioning utilities.

The audio and text branches must always be loaded from the same full CLAP
checkpoint.  Keeping that invariant in one class prevents a subtle but serious
failure mode where training and inference embeddings have the same shape but do
not live in the same representation space.
"""

import torch
import torch.nn.functional as F


DEFAULT_CLAP_MODEL = "laion/clap-htsat-unfused"


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class FrozenCLAPConditioner:
    """Frozen Hugging Face CLAP model exposing its two projected embeddings."""

    def __init__(
        self,
        model_name_or_path=DEFAULT_CLAP_MODEL,
        revision=None,
        device="auto",
        local_files_only=False,
    ):
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise ImportError(
                "CLAP conditioning requires transformers with ClapModel support. "
                "Install this project's requirements in the training environment."
            ) from exc

        self.device = resolve_device(device)
        load_kwargs = {"local_files_only": local_files_only}
        if revision:
            load_kwargs["revision"] = revision

        self.processor = ClapProcessor.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        self.model = ClapModel.from_pretrained(
            model_name_or_path, **load_kwargs
        ).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.resolved_revision = (
            getattr(self.model.config, "_commit_hash", None) or revision
        )

    @property
    def sample_rate(self):
        return int(self.processor.feature_extractor.sampling_rate)

    @property
    def audio_truncation(self):
        return self.processor.feature_extractor.truncation

    @property
    def audio_padding(self):
        return self.processor.feature_extractor.padding

    @staticmethod
    def _to_device(batch):
        return {
            key: value
            for key, value in batch.items()
            if torch.is_tensor(value)
        }

    @staticmethod
    def _validate(features):
        if features.ndim != 2:
            raise RuntimeError(
                f"CLAP must return [batch, embedding_dim], got {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all():
            raise RuntimeError("CLAP produced NaN or infinite embeddings")
        if torch.any(features.norm(p=2, dim=-1) <= 1e-12):
            raise RuntimeError("CLAP produced a zero-norm embedding")
        # Current ClapModel versions already normalize get_*_features().  Apply
        # it explicitly as a stable contract across transformers versions.
        return F.normalize(features.float(), p=2, dim=-1)

    @torch.inference_mode()
    def encode_audio(self, waveforms, max_length=None):
        processor_kwargs = dict(
            audios=waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        if max_length is not None:
            processor_kwargs["max_length"] = int(max_length)
        inputs = self._to_device(self.processor(**processor_kwargs))
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        audio_inputs = {
            key: value
            for key, value in inputs.items()
            if key in ("input_features", "is_longer")
        }
        return self._validate(self.model.get_audio_features(**audio_inputs))

    @torch.inference_mode()
    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self._to_device(
            self.processor(text=list(texts), return_tensors="pt", padding=True)
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        text_inputs = {
            key: value
            for key, value in inputs.items()
            if key in ("input_ids", "attention_mask")
        }
        return self._validate(self.model.get_text_features(**text_inputs))
