"""Deterministic JSONL text dataset for legal/open training corpora."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class JsonlDataset(Dataset):
    def __init__(self, path, seq_len, tokenizer):
        self.rows = [json.loads(x)["text"] for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.samples = []
        for text in self.rows:
            ids = tokenizer.encode(text)
            for i in range(0, max(1, len(ids) - 1), seq_len):
                chunk = ids[i:i+seq_len+1]
                if len(chunk) == seq_len + 1:
                    self.samples.append(torch.tensor(chunk, dtype=torch.long))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x = self.samples[idx]
        return x[:-1], x[1:]
