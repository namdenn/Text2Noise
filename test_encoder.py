import os
import torch
import torch.nn as nn
import json
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

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
        x = outputs.pooler_output
        x = self.mlp(x)
        return x


def encode_jsonl_dataset(input_jsonl, checkpoint_path, output_jsonl):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RobertaMLPEncoder(checkpoint_path=checkpoint_path)
    model.to(device)
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    with open(input_jsonl, 'r') as f:
        lines = f.readlines()

    print(f"Encoding {len(lines)} text entries...")
    with open(output_jsonl, 'w') as f_out:
        with torch.no_grad():
            for line in tqdm(lines):
                item = json.loads(line)
                text = item['text']

                inputs = tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding="max_length", 
                    truncation=True, 
                    max_length=32
                ).to(device)

                embedding = model(inputs['input_ids'], inputs['attention_mask'])
                
                item['embedding'] = embedding.squeeze(0).cpu().numpy().tolist()

                f_out.write(json.dumps(item) + '\n')

    print(f"saved to: {output_jsonl}")

if __name__ == "__main__":
    CKPT = os.environ.get(
        "TEXT_ENCODER_CHECKPOINT",
        "checkpoints/roberta_mlp_text_encoder.pt",
    )
    INPUT = "./metadata/train.jsonl"
    OUTPUT = "./metadata/train_encoded.jsonl"

    encode_jsonl_dataset(INPUT, CKPT, OUTPUT)
