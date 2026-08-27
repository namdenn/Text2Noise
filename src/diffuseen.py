#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from sgmse import sampling
from sgmse.model import ScoreModel
from sgmse.data_module import SpecsDataModule
from . import InferenceAlgoRegistry


@InferenceAlgoRegistry.register("diffuseen")
class DiffUSEEN:
    def __init__(
        self,
        ckpt_path="data/checkpoints/diffusion_gen_nonlinear_transform.ckpt",
        num_E=30,
        transform_type="exponent",
        eps=0.03,
        snr=0.5,
        sr=16000,
        verbose=False,
        device="cuda",  
    ):
        """
        Text-to-Noise Prior Sampling with FiLM Conditioning.

        Args:
            ckpt_path: Path to the pre-trained diffusion model.
            num_E: Number of iterations for the E step (reverse diffusion process).
            verbose: Whether to print progress information.
        """

        self.snr = snr
        self.sr = sr
        self.num_E = num_E
        self.verbose = verbose
        self.device = device
        self.eps = eps

        if self.verbose:
            print(f"Loading Text-to-Noise checkpoint: {ckpt_path}")

        # ==== Prior model ====
        self.model = ScoreModel.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
        )

        saved_data_config = dict(getattr(self.model, "_data_module_hparams", {}) or {})
        nested_data_config = saved_data_config.get("kwargs", {})
        if isinstance(nested_data_config, dict):
            saved_data_config = {**nested_data_config, **saved_data_config}
        preprocessing_keys = {
            "n_fft",
            "hop_length",
            "num_frames",
            "spec_factor",
            "spec_abs_exponent",
        }
        preprocessing_config = {
            key: value
            for key, value in saved_data_config.items()
            if key in preprocessing_keys
        }
        self.model.data_module = SpecsDataModule(
            train_jsonl="",
            val_jsonl="",
            test_jsonl="",
            batch_size=1,
            num_workers=0,
            transform_type=transform_type,
            **preprocessing_config,
        )
        self.model.eval()
        self.model.to(self.device)

        # Sampling and score normalization must use the same SDE.  Previously,
        # inference evolved samples with a separately hard-coded OUVESDE while
        # ScoreModel.forward() normalized scores with the checkpoint SDE.
        self.sde = self.model.sde.copy()
        self.sde.N = num_E
        self.NF = 1

    def to_audio(self, specto, total_samples):
        """
        Converts processed complex spectrogram back to waveforms using 
        the updated SpecsDataModule methods natively.
        """
        specto = specto * self.NF
        spec = specto.cpu().squeeze()

        # Call your updated data_module spec_back method directly
        spec = self.model.data_module.spec_back(spec)

        # Reconstruct waveform via your data module's native istft function
        waveform = self.model.data_module.istft(spec, length=total_samples)
            
        return waveform.cpu().reshape(1, -1)
    
    def predictor_corrector(self, St, t, text_condition, laststep, dt):
        """
        Processes a reverse diffusion step utilizing pre-computed 
        RoBERTa-MLP conditioning feature tensors.
        """
        with torch.no_grad():
    
            txt_emb = self._prepare_condition(text_condition, St.shape[0])

            # One annealed-Langevin corrector step.  Keep this method for
            # callers that use it directly; prior_sampler uses the shared PC
            # sampler below so training/inference utilities cannot drift.
            score = self.model.forward(St, t, txt_emb)
            std = self.sde.marginal_prob(St, t)[1]
            step_size = (self.snr * std) ** 2 * 2
            z = torch.randn_like(St)
            St_mean = St + step_size[:, None, None, None] * score
            St = St_mean + torch.sqrt(step_size * 2)[:, None, None, None] * z

            # Euler-Maruyama predictor for the reverse-time SDE.
            f, g = self.sde.sde(St, t)
            score = self.model.forward(St, t, txt_emb)
            z = torch.zeros_like(St) if laststep else torch.randn_like(St)
            St_mean = St - f * dt + (g**2)[:, None, None, None] * score * dt
            St = St_mean + g[:, None, None, None] * torch.sqrt(dt) * z

        return St_mean if laststep else St

    def _prepare_condition(self, condition, batch_size):
        if not torch.is_tensor(condition):
            raise TypeError(
                "condition must be a pre-computed text embedding tensor; "
                "use the same text encoder checkpoint used to build the training metadata"
            )
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        if condition.ndim != 2:
            raise ValueError(
                f"condition must have shape [batch, embedding_dim], got {tuple(condition.shape)}"
            )
        if condition.shape[0] == 1 and batch_size > 1:
            condition = condition.expand(batch_size, -1)
        elif condition.shape[0] != batch_size:
            raise ValueError(
                f"condition batch size {condition.shape[0]} does not match sample batch size {batch_size}"
            )
        if not torch.isfinite(condition).all():
            raise ValueError("condition contains NaN or infinite values")
        return condition.to(self.device, dtype=torch.float32)

    def prior_sampler(self, condition=None, duration_sec=5.0):
        """
        Generates structured noise profiles from scratch out of raw random seeds.
        """
        if duration_sec <= 0:
            raise ValueError(f"duration_sec must be positive, got {duration_sec}")

        condition = self._prepare_condition(condition, batch_size=1)
        window_length = self.model.data_module.n_fft
        freq_bins_stft = 1 + window_length // 2 

        total_samples = int(duration_sec * self.sr)
        hop_length = self.model.data_module.hop_length
        # Training uses torch.stft(center=True), for which a signal of length L
        # has 1 + floor(L / hop_length) frames.  The previous no-center formula
        # created a different grid and then changed the requested duration.
        nb_stft_frame = 1 + total_samples // hop_length

        downsampling_factor = 2 ** (self.model.dnn.num_resolutions - 1)
        if nb_stft_frame % downsampling_factor != 0:
            nb_stft_frame = (
                (nb_stft_frame // downsampling_factor) + 1
            ) * downsampling_factor

        if self.verbose:
            print(
                "Condition embedding: "
                f"shape={tuple(condition.shape)}, "
                f"mean={condition.mean().item():.4f}, "
                f"std={condition.std().item():.4f}"
            )
            print(f"Target Grid Shape: [{freq_bins_stft} bins x {nb_stft_frame} frames]")

        sample_shape = (1, 1, freq_bins_stft, nb_stft_frame)
        shape_reference = torch.empty(
            sample_shape, dtype=torch.cfloat, device=self.device
        )

        def score_fn(x, t):
            return self.model.forward(x, t, condition)

        pc_sampler = sampling.get_pc_sampler(
            predictor_name="euler_maruyama",
            corrector_name="ald",
            sde=self.sde,
            score_fn=score_fn,
            y=shape_reference,
            denoise=True,
            eps=self.eps,
            snr=self.snr,
            corrector_steps=1,
        )
        St, _ = pc_sampler()

        st = self.to_audio(St, total_samples)
        St_spec = St.squeeze().cpu()

        return st.squeeze().numpy(), St_spec
