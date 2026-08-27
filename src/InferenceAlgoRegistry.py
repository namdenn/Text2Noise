import os

from . import InferenceAlgoRegistry
from joint_paradiffuseen import jointparadiffuseen
from utils import show_spec
import IPython.display

clean_file = os.environ.get("CLEAN_AUDIO", "data/test_speech/s.wav")
vfile_path = os.environ.get("VIDEO_FEATURES", "data/test_speech/x.wav")
ckpt_path = os.environ.get(
    "DIFFUSION_CHECKPOINT",
    "checkpoints/diffusion_gen_nonlinear_transform.ckpt",
)
nun_E = 30
verbose = True

jointparadiffuseen = jointparadiffuseen(ckpt_path=ckpt_path, num_E=num_E, verbose=verbose)
JointParaDiffUSEEN = InferenceAlgoRegistry.get_by_name("joint_paradiffuseen")
s_clean, S_clean = jointparadiffuseen.prior_sampler(clean_file, vfile_path,noise=False)

show_spec(spectogram=[S_clean], titles=["Sampled speech spectrogram"])
IPython.display.display(IPython.display.Audio(s_clean, rate=16000))


n_noise, N_noise = jointparadiffuseen.prior_sampler(clean_file, vfile_path,noise=True)

show_spec(spectogram=[N_noise], titles=["Sampled noise spectrogram"])
IPython.display.display(IPython.display.Audio(n_noise, rate=16000))
