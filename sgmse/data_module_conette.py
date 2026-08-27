import os
import json
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchaudio import load
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

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

    def forward(self, x): 
        return self.sequential(x)

class RobertaMLPEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.mlp = MLPLayers(units=[768, 512, 512])
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.mlp(outputs.pooler_output)

def encode_metadata_offline(input_dir, output_dir):
    """
    Reads JSON/JSONL files with existing text captions, encodes them with 
    RoBERTa embeddings, and saves the output into a new folder.
    Handles both multi-line JSON objects and single-line JSONL formats automatically.
    """
    ckpt = os.environ.get(
        "TEXT_ENCODER_CHECKPOINT",
        "checkpoints/text_encoder_only.pt",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading encoder model on {device}...")
    model = RobertaMLPEncoder(checkpoint_path=ckpt).to(device).eval()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = [
        "conette_train.jsonl",
        "conette_val.jsonl",
        "conette_test.jsonl"
    ]

    for filename in files_to_process:
        in_p = os.path.join(input_dir, filename)
        out_p = os.path.join(output_dir, filename)
        
        if not os.path.exists(in_p):
            print(f"Warning: File {in_p} does not exist, skipping...")
            continue
        
        print(f"Loading entries from {in_p}...")
        
        items = []
        with open(in_p, 'r', encoding='utf-8') as f_in:
            content = f_in.read().strip()
            
            try:
                items = json.loads(content)
                if isinstance(items, dict):
                    items = [items]
            except json.JSONDecodeError:
                decoder = json.JSONDecoder()
                idx = 0
                while idx < len(content):
                    while idx < len(content) and content[idx].isspace():
                        idx += 1
                    if idx >= len(content):
                        break
                    obj, end_idx = decoder.raw_decode(content, idx)
                    items.append(obj)
                    idx = end_idx

        print(f"Encoding {len(items)} items from {filename} -> {out_p}...")
        
        with open(out_p, 'w', encoding='utf-8') as f_out:
            for item in tqdm(items):
                with torch.no_grad():
                    tok = tokenizer(
                        item['text'], 
                        return_tensors="pt", 
                        padding="max_length", 
                        truncation=True, 
                        max_length=32
                    ).to(device)
                    
                    embed = model(tok['input_ids'], tok['attention_mask'])
                    item['embedding'] = embed.squeeze(0).cpu().numpy().tolist()
                
                f_out.write(json.dumps(item) + '\n')
                
    print(f"\nSuccessfully encoded metadata and saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    INPUT_METADATA_DIR = os.environ.get(
        "METADATA_INPUT_DIR",
        "generated/metadata/input",
    )
    
    OUTPUT_ENCODED_DIR = os.environ.get(
        "METADATA_OUTPUT_DIR",
        "generated/metadata/encoded",
    )
    
    encode_metadata_offline(
        input_dir=INPUT_METADATA_DIR, 
        output_dir=OUTPUT_ENCODED_DIR
    )
