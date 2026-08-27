import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchaudio import load
import numpy as np
import json
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
from tqdm import tqdm


# Training and inference must use identical text-encoder weights.  The
# environment override keeps the cluster-specific location configurable while
# providing one shared source of truth for both paths.
DEFAULT_TEXT_ENCODER_CHECKPOINT = os.environ.get(
    "TEXT_ENCODER_CHECKPOINT",
    "checkpoints/text_encoder_only.pt",
)

class MLPLayers(nn.Module):
    def __init__(self, units=[768, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(nonlin)
            sequence.append(nn.Dropout(dropout))
        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)
    def forward(self, x): return self.sequential(x)

class RobertaMLPEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.mlp = MLPLayers(units=[768, 512, 512])
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict):
                for container_key in ("state_dict", "model_state_dict"):
                    if container_key in checkpoint and isinstance(
                        checkpoint[container_key], dict
                    ):
                        checkpoint = checkpoint[container_key]
                        break
            if not isinstance(checkpoint, dict):
                raise TypeError(
                    f"Unsupported text encoder checkpoint format: {type(checkpoint).__name__}"
                )

            # Accept common wrapper prefixes (for example "module." or
            # "text_encoder.") but never silently continue with a randomly
            # initialized MLP when no compatible weights were found.
            model_keys = set(self.state_dict())
            state_dict = {}
            for checkpoint_key, value in checkpoint.items():
                candidates = [checkpoint_key, f"mlp.{checkpoint_key}"]
                candidates.extend(
                    checkpoint_key[index + 1 :]
                    for index, char in enumerate(checkpoint_key)
                    if char == "."
                )
                matching_key = next(
                    (candidate for candidate in candidates if candidate in model_keys),
                    None,
                )
                if matching_key is not None:
                    state_dict[matching_key] = value

            required_mlp_keys = {key for key in model_keys if key.startswith("mlp.")}
            missing_mlp_keys = sorted(required_mlp_keys - state_dict.keys())
            if missing_mlp_keys:
                raise RuntimeError(
                    "The text encoder checkpoint did not provide all MLP weights. "
                    f"Missing keys: {missing_mlp_keys}. This would make training and "
                    "inference embeddings inconsistent."
                )
            self.load_state_dict(state_dict, strict=False)
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.mlp(outputs.pooler_output)

def create_split_jsonl(base_path, output_dir="./metadata"):
    """ Scans folders and creates raw JSONL files with dynamic noise descriptions. """
    noise_name = os.path.basename(base_path.rstrip("/"))
    
    splits = ['train', 'test', 'val']
    os.makedirs(output_dir, exist_ok=True)
    
    for split in splits:
        split_folder = os.path.join(base_path, split)
        output_file = os.path.join(output_dir, f"{split}.jsonl")
        
        wav_files = sorted(glob(os.path.join(split_folder, "**/*.wav"), recursive=True))
        if not wav_files:
            print(f"Warning: No files found in {split_folder}")
            continue
            
        with open(output_file, 'w', encoding='utf-8') as f:
            for wav_path in wav_files:
                data_entry = {
                    "wav_path": os.path.abspath(wav_path),
                    "text": f"This is {noise_name} noise"
                }
                f.write(json.dumps(data_entry) + '\n')

def combine_files(source_paths, destination_path):
    """Helper function to cleanly concatenate multiple JSONL files."""
    if not source_paths:
        return
    
    print(f"Combining {len(source_paths)} files into -> {destination_path}")
    with open(destination_path, 'w', encoding='utf-8') as outfile:
        for path in source_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        if line.strip():
                            outfile.write(line.strip() + "\n")
            else:
                print(f"Warning: Expected file missing during combination: {path}")

def encode_metadata_offline(input_dir="./metadata", output_dir="./metadata_encoded"):
    """ Takes raw JSONL and adds RoBERTa embeddings. """
    ckpt = DEFAULT_TEXT_ENCODER_CHECKPOINT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RobertaMLPEncoder(checkpoint_path=ckpt).to(device).eval()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    os.makedirs(output_dir, exist_ok=True)

    for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
        in_p, out_p = os.path.join(input_dir, split), os.path.join(output_dir, split)
        if not os.path.exists(in_p): continue
        
        print(f"Encoding {split} to embeddings...")
        with open(in_p, 'r') as f_in, open(out_p, 'w') as f_out:
            for line in tqdm(f_in):
                item = json.loads(line)
                with torch.no_grad():
                    tok = tokenizer(item['text'], return_tensors="pt", padding="max_length", truncation=True, max_length=32).to(device)
                    embed = model(tok['input_ids'], tok['attention_mask'])
                    item['embedding'] = embed.squeeze(0).cpu().numpy().tolist()
                f_out.write(json.dumps(item) + '\n')
    print(f"Final Encoded metadata saved to: {output_dir}")

class Specs(Dataset):
    def __init__(self, jsonl_path, num_frames=256, normalize="clean", spec_transform=None, stft_kwargs=None, dummy=False, **kwargs):
        super().__init__()
        self.samples = [json.loads(line) for line in open(jsonl_path, 'r')]
        self.num_frames, self.normalize = num_frames, normalize
        self.spec_transform, self.stft_kwargs = spec_transform, stft_kwargs
        self.hop_length = stft_kwargs["hop_length"]
        self.dummy = dummy
        self.shuffle_spec = kwargs.get("shuffle_spec", True)

    def __getitem__(self, i):
        sample = self.samples[i]
        x, _ = load(sample['wav_path'])
        t_len = (self.num_frames - 1) * self.hop_length
        if x.size(-1) > t_len:
            start = int(np.random.uniform(0, x.size(-1) - t_len)) if self.shuffle_spec else int((x.size(-1)-t_len)/2)
            x = x[..., start : start + t_len]
        else:
            x = F.pad(x, (0, t_len - x.size(-1)))
        if self.normalize == "clean": x = x / (x.abs().max() + 1e-8)
        X = torch.stft(x, **self.stft_kwargs)
        if self.spec_transform: X = self.spec_transform(X)
        return X, torch.tensor(sample['embedding'], dtype=torch.float32)

    def __len__(self):
        return len(self.samples) // 100 if self.dummy else len(self.samples)

class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        group = parser.add_argument_group("DataModule")
        group.add_argument("--train_jsonl", type=str, required=True)
        group.add_argument("--val_jsonl", type=str, required=True)
        group.add_argument("--test_jsonl", type=str, required=True)
        group.add_argument("--batch_size", type=int, default=8)
        group.add_argument("--num_workers", type=int, default=4)
        group.add_argument("--n_fft", type=int, default=510)
        group.add_argument("--hop_length", type=int, default=128)
        group.add_argument("--num_frames", type=int, default=256)
        group.add_argument("--spec_factor", type=float, default=0.15)
        group.add_argument("--spec_abs_exponent", type=float, default=0.5)
        group.add_argument("--dummy", action="store_true")
        # group.add_argument("--conditioning_dim", type=int, default=512)
        return parser
    
    def __init__(self, train_jsonl, val_jsonl, test_jsonl, batch_size=8, num_workers=4, n_fft=510, hop_length=128, num_frames=256, dummy=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_jsonl, self.val_jsonl, self.test_jsonl = train_jsonl, val_jsonl, test_jsonl
        self.n_fft, self.hop_length, self.num_frames = n_fft, hop_length, num_frames
        self.batch_size, self.num_workers, self.dummy = batch_size, num_workers, dummy
        self.window = torch.hann_window(n_fft, periodic=True)
        self.windows = {}
        self.transform_type = kwargs.get("transform_type", "exponent")
        self.spec_factor = kwargs.get("spec_factor", 0.15)
        self.spec_abs_exponent = kwargs.get("spec_abs_exponent", 0.5)

    def spec_fwd(self, spec):
        spec = spec.abs()**self.spec_abs_exponent * torch.exp(1j * spec.angle())
        return spec * self.spec_factor

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "normalise":
            spec = spec
        elif self.transform_type == "none":
            spec = spec
        return spec
    
    @property
    def stft_kwargs(self):
        return {"n_fft": self.n_fft, "hop_length": self.hop_length, "window": self.window, "center": True, "return_complex": True}

    def setup(self, stage=None):
        k = dict(stft_kwargs=self.stft_kwargs, num_frames=self.num_frames, spec_transform=self.spec_fwd, dummy=self.dummy)
        if stage in ("fit", None):
            self.train_set = Specs(self.train_jsonl, **k, shuffle_spec=True)
            self.valid_set = Specs(self.val_jsonl, **k, shuffle_spec=False)
        if stage in ("test", None):
            self.test_set = Specs(self.test_jsonl, **k, shuffle_spec=False)

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(
            spec, **{**self.istft_kwargs, "window": window, "length": length}
        )
    
    def train_dataloader(self): return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    def val_dataloader(self): return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self): return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

if __name__ == "__main__":
    CORPUS_ROOT = os.environ.get("CORPUS_ROOT", "data/text2noise")
    
    noise_folders = ["babble", "car", "cafe", "street", "lr", "white"]
    
    BASE_METADATA_DIR = "./metadata_individual"
    COMBINED_RAW_DIR = "./metadata_combination"
    COMBINED_ENCODED_DIR = "./metadata_combination_encoded_audioldm_cpt"
    
    individual_train_paths = []
    individual_val_paths = []
    individual_test_paths = []

    for noise in noise_folders:
        data_base_path = os.path.join(CORPUS_ROOT, noise)
        output_dir = os.path.join(BASE_METADATA_DIR, f"metadata_{noise}")
        
        if not os.path.exists(data_base_path):
            print(f"Error: Core directory not found for {noise}, skipping...")
            continue
            
        create_split_jsonl(data_base_path, output_dir=output_dir)
        
        individual_train_paths.append(os.path.join(output_dir, "train.jsonl"))
        individual_val_paths.append(os.path.join(output_dir, "val.jsonl"))
        individual_test_paths.append(os.path.join(output_dir, "test.jsonl"))

    os.makedirs(COMBINED_RAW_DIR, exist_ok=True)
    
    combine_files(individual_train_paths, os.path.join(COMBINED_RAW_DIR, "train.jsonl"))
    combine_files(individual_val_paths, os.path.join(COMBINED_RAW_DIR, "val.jsonl"))
    combine_files(individual_test_paths, os.path.join(COMBINED_RAW_DIR, "test.jsonl"))
    
    encode_metadata_offline(input_dir=COMBINED_RAW_DIR, output_dir=COMBINED_ENCODED_DIR)
    
