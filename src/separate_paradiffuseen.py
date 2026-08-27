#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import warnings

import torch
import numpy as np
from tqdm import tqdm
from src.utils import LinearScheduler, calc_metrics
from sgmse.model import ScoreModel
from torchaudio import load
from sgmse.util.other import pad_spec
from sgmse.util.utils_video import load_array, resample_video, prep_video, videocap
from sgmse.data_module import SpecsDataModule
from . import InferenceAlgoRegistry


class _IgnoreMissingTextCondition(torch.nn.Module):
    """Keep legacy speech priors on their original unconditioned score path."""

    def forward(self, _condition):
        return 0.0


def _prepare_legacy_speech_checkpoint(checkpoint_path):
    """Return a loadable temporary copy when a speech checkpoint predates text conditioning."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Speech checkpoint has no Lightning state_dict: {checkpoint_path}"
        )

    weight_key = "dnn.text_embedding.weight"
    bias_key = "dnn.text_embedding.bias"
    has_weight = weight_key in state_dict
    has_bias = bias_key in state_dict
    if has_weight != has_bias:
        raise RuntimeError(
            "Speech checkpoint contains an incomplete text_embedding layer: "
            f"weight={has_weight}, bias={has_bias}"
        )
    if has_weight:
        return checkpoint_path, None, False

    hyper_parameters = checkpoint.get("hyper_parameters", {})
    if not isinstance(hyper_parameters, dict):
        hyper_parameters = {}
    nested = hyper_parameters.get("kwargs", {})
    if not isinstance(nested, dict):
        nested = {}
    conditioning_fusion = str(
        hyper_parameters.get(
            "conditioning_fusion",
            nested.get("conditioning_fusion", "additive"),
        )
    ).lower()
    if conditioning_fusion != "additive":
        raise RuntimeError(
            "Legacy speech compatibility currently requires additive conditioning, "
            f"but the checkpoint requests {conditioning_fusion!r}"
        )

    # The first timestep MLP bias has size 4*nf, exactly the output size of the
    # additive text projection introduced after this speech model was trained.
    timestep_bias = state_dict.get("dnn.all_modules.1.bias")
    if timestep_bias is not None:
        projection_dim = int(timestep_bias.numel())
    else:
        nf = int(hyper_parameters.get("nf", nested.get("nf", 96)))
        projection_dim = 4 * nf
    conditioning_dim = int(
        hyper_parameters.get(
            "conditioning_dim",
            nested.get("conditioning_dim", 512),
        )
    )
    state_dict[weight_key] = torch.zeros(projection_dim, conditioning_dim)
    state_dict[bias_key] = torch.zeros(projection_dim)

    # Its EMA shadow list also predates the compatibility parameters. Removing
    # EMA only from the temporary copy makes ScoreModel use regular weights.
    checkpoint.pop("ema", None)
    handle, temporary_path = tempfile.mkstemp(
        prefix="speech_checkpoint_compat_",
        suffix=".ckpt",
    )
    os.close(handle)
    try:
        torch.save(checkpoint, temporary_path)
    except Exception:
        os.unlink(temporary_path)
        raise
    warnings.warn(
        "Loading a legacy unconditioned speech checkpoint through a temporary "
        f"compatibility copy with text projection shape "
        f"({projection_dim}, {conditioning_dim}). The original checkpoint is unchanged."
    )
    return temporary_path, temporary_path, True

@InferenceAlgoRegistry.register("separate_paradiffuseen")
class SeparateParaDiffUSEEN:
    @staticmethod
    def _attach_data_module(model, transform_type):
        saved_config = dict(getattr(model, "_data_module_hparams", {}) or {})
        nested_config = saved_config.get("kwargs", {})
        if isinstance(nested_config, dict):
            saved_config = {**nested_config, **saved_config}
        preprocessing_keys = {
            "n_fft",
            "hop_length",
            "num_frames",
            "spec_factor",
            "spec_abs_exponent",
        }
        preprocessing_config = {
            key: value
            for key, value in saved_config.items()
            if key in preprocessing_keys
        }
        model.data_module = SpecsDataModule(
            train_jsonl="",
            val_jsonl="",
            test_jsonl="",
            batch_size=1,
            num_workers=0,
            transform_type=transform_type,
            **preprocessing_config,
        )

    def __init__(
        self,
        ckpt_path= "", 
        ckpt_noise="",
        num_E=30,
        transform_type="exponent",
        delta=1e-10,
        eps=0.03,
        snr=0.5,
        sr=16000,
        verbose=False,
        listen_noise = False,
        device= "cuda",
        print_metrics = False,
        set_v_to_zero = "no",
        optimized_lambda= False,
    ):
        self.snr = snr
        self.sr = sr
        self.delta = delta
        self.num_E = num_E
        self.verbose = verbose
        self.listen_noise = listen_noise
        self.device = device

        self.scheduler = LinearScheduler(N=num_E, eps=eps)
        
        # ==== For prior speech model ====
        speech_checkpoint, temporary_checkpoint, ignore_text_condition = (
            _prepare_legacy_speech_checkpoint(ckpt_path)
        )
        try:
            self.model = ScoreModel.load_from_checkpoint(
                speech_checkpoint,
                map_location="cpu",
            )
            self._attach_data_module(self.model, transform_type)
            if ignore_text_condition:
                self.model.dnn.text_embedding = _IgnoreMissingTextCondition()
            self.sde = self.model.sde.copy()
            self.sde.N = num_E
            self.model.eval(no_ema=False)
            self.model.to(self.device)

            # CPU copy is used for STFT/iSTFT on the gres cluster.
            self.model_cpu = ScoreModel.load_from_checkpoint(
                speech_checkpoint,
                map_location="cpu",
            )
            self._attach_data_module(self.model_cpu, transform_type)
            if ignore_text_condition:
                self.model_cpu.dnn.text_embedding = _IgnoreMissingTextCondition()
            self.model_cpu.eval(no_ema=False)
        finally:
            if temporary_checkpoint is not None:
                os.unlink(temporary_checkpoint)

        self.audio_only = getattr(
            self.model,
            "audio_only",
            getattr(self.model.dnn, "audio_only", True),
        )
        if not self.audio_only: 
            self.fps = 30
            self.video_feature_type = self.model.dnn.video_feature_type
            self.vfeat_processing_order = self.model.dnn.vfeat_processing_order
            self.set_v_to_zero = set_v_to_zero
        else: 
            self.vfeat_processing_order = "default"  
        
        # ==== For prior noise model ====        
        self.model_noise = ScoreModel.load_from_checkpoint(
            ckpt_noise,
            map_location="cpu",
        )
        self._attach_data_module(self.model_noise, transform_type)
        self.sde_noise = self.model_noise.sde.copy()
        self.sde_noise.N = num_E
        self.model_noise.eval(no_ema=False)
        self.model_noise.to(self.device)

        self.print_metrics = print_metrics

    def pick_zeta_schedule(self, schedule, t, zeta, linear_t=None, clip=50_000, max_step=0.9, decay_rate=1.0, increase_rate=1.0):
        if schedule == "none": return None
        if schedule == "constant": zeta_t = zeta
        if schedule == "lin-decrease": zeta_t = zeta * t
        if schedule == "lin-increase": zeta_t = zeta * (1 - t)
        if schedule == "half-cycle": zeta_t = zeta * np.sin(np.pi * t)
        if schedule == "sqrt-increase": zeta_t = zeta * np.sqrt(1e-10 + t)
        if schedule == "exp-increase": zeta_t = zeta * np.exp(t)
        if schedule == "exp-decrease": zeta_t = zeta * (np.exp(increase_rate * linear_t / max_step) - 1) / (np.exp(increase_rate) - 1)
        if schedule == "log-increase": zeta_t = zeta * np.log(1 + 1e-10 + t)
        if schedule == "div-sig": zeta_t = zeta / t
        if schedule == "div-sig-square": zeta_t = zeta / t**2
        if schedule == "sigma": zeta_t = zeta * self.sde._std(t)
        if schedule == "sigma_like":
            zeta_min, zeta_max, theta = 1e-5, 0.5, 1.5
            logsig = np.log(zeta_max / zeta_min)
            zeta_t = zeta * torch.sqrt((zeta_min**2 * torch.exp(-2 * theta * t) * (torch.exp(2 * (theta + logsig) * t) - 1) * logsig) / (theta + logsig))
        if schedule == "saw-tooth-increase":
            zeta_t = zeta / max_step * linear_t if linear_t < max_step else zeta + zeta * (max_step - linear_t) / (1 - max_step)
        if schedule == "saw-tooth-exp":
            zeta_t = zeta * (np.exp(increase_rate * linear_t / max_step) - 1) / (np.exp(increase_rate) - 1) if linear_t < max_step else zeta * np.exp(-decay_rate * (linear_t - max_step) / (1 - max_step))
        return min(zeta_t, clip)

    def load_visual_data(self, vfile_path):   
        if self.vfeat_processing_order in ["cut_extract"]:            
            video_size_dict = {"avhubert":88,"resnet":88, "raw_image":88, "flow_avse":112}
            v = prep_video(video_path=vfile_path, start_frame=0, video_size=video_size_dict[self.video_feature_type], video_feature_type=self.video_feature_type)     
            v = v.to(self.device) 
            nb_v_frame = v.shape[0] if self.video_feature_type in ["flow_avse"] else v.shape[1]
        return v, nb_v_frame
    
    def load_data(self, file_path, add_noise=False, vfile_path = None):
        x, sr = load(file_path)
        if add_noise:
            x += 1e-4*torch.randn_like(x)
        assert sr == self.sr
        self.T_orig = x.size(1)

        X = pad_spec(torch.unsqueeze(self.model_cpu._forward_transform(self.model_cpu._stft(x)), 0)).to(self.device)  

        if not self.audio_only:
            assert vfile_path is not None
            if self.vfeat_processing_order in ["cut_extract"]:
                v, _ = self.load_visual_data(vfile_path)                   
        else:
            v = None          
        return x, X, v 

    def to_audio(self, specto):
        return self.model.to_audio(specto.squeeze(), self.T_orig).cpu().reshape(1, -1)

    def to_audio_tr(self, specto): 
        specto = specto.cpu()
        return self.model_cpu._istft(specto, self.T_orig).cpu().reshape(1, -1)    

    def predictor_corrector(self, St, condition, t, laststep, dt, noise=False):
        if not noise: 
            score_model = self.model
            sde = self.sde
        else: 
            score_model = self.model_noise
            sde = self.sde_noise

        with torch.no_grad():            
            std = sde.marginal_prob(St, t)[1]

        with torch.no_grad():
            f, g = sde.sde(St, t)
            score = score_model.forward(St, t, condition)
            z = torch.zeros_like(St) if laststep else torch.randn_like(St)
            St = (
                St
                - f * dt
                + (g**2)[:, None, None, None] * score * dt
                + g[:, None, None, None] * torch.sqrt(dt) * z
            )
            torch.cuda.empty_cache()

        return St, std, score, g

    def likelihood_update_individual(
        self, St, N0, t, std, dt, lmbd, noise=False, w_up=True
    ):
        with torch.no_grad():
            sde = self.sde_noise if noise else self.sde
            theta = sde.theta
            mu_t = torch.exp(-theta * t)[:, None, None, None]
            _, g = sde.sde(St, t)

            difference = self.X - (St / mu_t + N0) 
            w = 8e-3 
            nppls = ((1 / mu_t) * difference / ((std[:, None, None, None] / mu_t) ** 2 + w)).type(torch.complex64)

            weight = lmbd * (g**2)[:, None, None, None]
            St = St + weight * nppls * dt
            return St 

    def prior_sampler(self, clean_file = None, vfile_path = None, text_embedding=None, noise=False):
        self.prior_sampling = True
        score_model = self.model if not noise else self.model_noise
        timesteps = self.scheduler.timesteps()
        self.NF = 1

        window_length = self.model.data_module.n_fft
        freq_bins_stft = 1 + window_length // 2

        if (self.audio_only == True and noise == False) or (noise == True):
            self.T_orig = 80000
            nb_stft_frame = 640
            v = None
        else:
            assert vfile_path is not None 
            assert clean_file is not None 
            audio, spec, v = self.load_data(file_path=clean_file, vfile_path=vfile_path)
            v = v.unsqueeze(dim=0) 
            self.T_orig = audio.size(1)
            nb_stft_frame = spec.shape[-1]

        St = torch.randn(1, 1, freq_bins_stft, nb_stft_frame, dtype=torch.cfloat, device=self.device) * self.sde._std(torch.ones(1, device=self.device))
        dt = torch.tensor(1 / self.num_E, device=self.device)

        for i in tqdm(range(0, self.num_E)):
            t = torch.tensor([timesteps[i]], device=self.device)
            current_cond = text_embedding if noise else v
            St, _, _, _ = self.predictor_corrector(
                St=St,                
                t=t,
                condition=current_cond,
                laststep=i == (self.num_E - 1),
                dt=dt,
                noise=noise,
            )

        st = self.to_audio(St)
        St = score_model._backward_transform(St)
        return st, St
        
    def posterior_sampler(self, text_embedding=None, startstep=0, S0=None, N0=None):  
        timesteps = self.scheduler.timesteps()
        self.prior_sampling = False

        t = torch.tensor([timesteps[startstep]], device=self.device).repeat(self.nbatch)
        S_mean, _ = self.sde.marginal_prob(self.X, t)

        if S0 is None:
            St = torch.randn_like(self.X) * self.sde._std(timesteps[startstep]*torch.ones(1, device=self.device)) + S_mean
        else:
            St = torch.randn_like(self.X) * self.sde._std(timesteps[startstep]*torch.ones(1, device=self.device)) + S0          

        N_mean, _ = self.sde_noise.marginal_prob(St - self.X, t)

        if N0 is None:
            Nt = torch.randn_like(self.X) * self.sde_noise._std(timesteps[startstep]*torch.ones(1, device=self.device)) + N_mean
        else:
            Nt = torch.randn_like(self.X) * self.sde_noise._std(timesteps[startstep]*torch.ones(1, device=self.device)) + N0

        dt = torch.tensor(1 / self.num_E, device=self.device)
        range_i = tqdm(range(startstep, self.num_E)) if self.verbose else range(startstep, self.num_E)

        S0hat = S0 if S0 is not None else St
        N0hat = N0 if N0 is not None else Nt        

        for i in range_i:
            t = torch.tensor([timesteps[i]], device=self.device).repeat(self.nbatch)
            
            # Update Speech (Conditioned on Video)
            St, std, S_score, _ = self.predictor_corrector(
                St=St,                                
                t=t,
                condition=self.visual_feature,
                laststep=i == (self.num_E - 1),
                dt=dt,
                noise=False
            )

            # Update Noise (Conditioned on Text Embedding)
            Nt, std_noise, N_score, _ = self.predictor_corrector(
                St=Nt,                 
                t=t,
                condition=text_embedding,
                laststep=i == (self.num_E - 1),
                dt=dt,
                noise=True
            )

            lmbd = self.pick_zeta_schedule(
                schedule="sigma",
                t=torch.tensor([timesteps[i]], device=self.device),
                zeta=self.lmbd,
                linear_t=(self.num_E - i) / self.num_E,
                max_step=0.99,
            )

            if i % self.project_every_k_steps == 0 and i < self.num_E - 1:
                gamma_speech_t = torch.exp(-self.sde.theta * t)[:, None, None, None]
                gamma_noise_t = torch.exp(-self.sde_noise.theta * t)[
                    :, None, None, None
                ]
                N0hat = (
                    Nt + std_noise.square()[:, None, None, None] * N_score
                ) / gamma_noise_t
                S0hat = (
                    St + std.square()[:, None, None, None] * S_score
                ) / gamma_speech_t

                St = self.likelihood_update_individual(
                    St=St,
                    N0=N0hat,
                    t=t,
                    std=std,
                    dt=dt,
                    lmbd=lmbd,
                )
                Nt = self.likelihood_update_individual(
                    St=Nt,
                    N0=S0hat,
                    t=t,
                    std=std_noise,
                    dt=dt,
                    lmbd=lmbd,
                    noise=True,
                )

        return St, Nt

    def run(
        self,
        mix_file,
        clean_file=None,  
        video_file = None,   
        text_embedding = None, 
        lmbd=5.75,
        nbatch=8,
        num_EM=1,
        nmf_rank=4,
        project_every_k_steps=1,
        std_measurement = 0.15,       
        startstep=0,
        wiener_filter=True,
        mixture_consistency=False,
        refine=False,
        S0=None,
        N0=None,        
    ):
        self.lmbd = lmbd
        self.project_every_k_steps = project_every_k_steps
        self.nbatch = nbatch
        self.std_measurement = std_measurement
        self.wiener_filter = wiener_filter

        x, X, v = self.load_data(file_path = mix_file, add_noise=True, vfile_path = video_file)
        self.x = x
        self.NF = X.abs().max()
        X = X / self.NF

        if self.verbose and clean_file != None:
            s_ref, S_ref, _  = self.load_data(file_path=clean_file, add_noise=False, vfile_path = video_file)
            self.s_ref = s_ref
            self.S_ref = S_ref
            s_ref = s_ref.numpy().reshape(-1)
                     
            x_withoutgaussian_noise, _, _  = self.load_data(file_path = mix_file, add_noise=False, vfile_path = video_file)
            x_withoutgaussian_noise = x_withoutgaussian_noise.numpy().reshape(-1)
            
            if self.print_metrics:
                metrix = calc_metrics(s_ref, x_withoutgaussian_noise, x_withoutgaussian_noise - s_ref)
                print(f"Input PESQ: {metrix['pesq']:.4f} --- SI-SDR: {metrix['si_sdr']:.4f}")

        if S0 is None:
            self.X = X.repeat(self.nbatch, 1, 1, 1)
        else:
            self.X = S0

        if not self.audio_only:        
            if self.vfeat_processing_order in ["cut_extract"]:
                if self.video_feature_type in ["resnet", "avhubert"]: 
                    self.visual_feature = v.repeat(self.nbatch, 1, 1, 1, 1)
                elif self.video_feature_type in ["flow_avse"]: 
                    self.visual_feature = v.repeat(self.nbatch, 1, 1, 1)
                elif self.video_feature_type in ["raw_image"]:              
                    self.visual_feature = v.repeat(self.nbatch, 1, 1)
        else: 
            self.visual_feature = None

        St, Nt = self.posterior_sampler(text_embedding=text_embedding, startstep=startstep, S0=S0, N0=N0)

        S0, N0 = St.clone(), Nt.clone()
        self.S0, self.N0 = S0, N0

        Nt_postprocess = N0.clone()

        if mixture_consistency:
            St_hat = self.model._backward_transform(St).mean(0) * self.NF
            Nt_hat = self.model._backward_transform(Nt).mean(0) * self.NF
            X_true = self.model._backward_transform(X).squeeze() * self.NF
            X_hat = St_hat + Nt_hat
            St = St_hat + 0.5*(X_true - X_hat)
            Nt = Nt_hat + 0.5*(X_true - X_hat)

        if self.wiener_filter:
            X = X * self.NF
            St_abs_2 = (S0.abs().pow(2)/(S0.abs().pow(2) + N0.abs().pow(2))).mean(0) * X.abs().pow(2)
            St = St_abs_2.sqrt() * torch.exp(1j * torch.angle(S0.mean(0)))
            
            Nt_abs_2 = (N0.abs().pow(2)/(S0.abs().pow(2) + N0.abs().pow(2))).mean(0) * X.abs().pow(2)
            Nt = Nt_abs_2.sqrt() * torch.exp(1j * torch.angle(N0.mean(0)))
            
            Nt_complex = Nt.clone()
            St = self.model._backward_transform(St).squeeze()
            Nt = self.model._backward_transform(Nt).squeeze()
            Nt_postprocess = Nt_complex 
            X = self.model._backward_transform(X).squeeze()
            self.St, self.Nt, self.Xt = St, Nt, X.squeeze()
            
        elif not mixture_consistency and not refine:
            St = self.model._backward_transform(St).mean(0) * self.NF

        if refine:
            St_hat = St.clone()
            St_abs = torch.maximum(self.X.abs() - Nt.abs(), torch.tensor(0.0))
            St = St_abs * torch.exp(1j * torch.angle(St_hat))
            St = self.model._backward_transform(St).squeeze().mean(0)

        st = self.to_audio_tr(St).numpy().reshape(-1)
        
        if not self.listen_noise:
            return st, St
        else:
            if not self.wiener_filter:
                Nt = self.model._backward_transform(Nt).mean(0) * self.NF
            nt = self.to_audio_tr(Nt).numpy().reshape(-1)

            St_postprocess = self.X.mean(0) - Nt_postprocess.mean(0)
            St_postprocess = self.model._backward_transform(St_postprocess).squeeze() * self.NF
            st_postprocess = self.to_audio_tr(St_postprocess).numpy().reshape(-1)                
                
            return st, St, nt, Nt, st_postprocess, St_postprocess
