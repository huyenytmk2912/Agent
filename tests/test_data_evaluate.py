import json
import torch
from torch.utils.data import DataLoader
from atlas.data import JsonlDataset
from atlas.evaluate import evaluate
from atlas.model import AtlasModel

class Tok:
    def encode(self, s): return [1] + list(s.encode()) + [2]

def cfg():
    return type("Cfg", (), dict(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=176, max_seq_len=8, rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True))

def test_jsonl_dataset(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(json.dumps({"text": "abcdefgh"}) + "\n", encoding="utf-8")
    ds = JsonlDataset(p, 4, Tok())
    assert len(ds) >= 1
    x, y = ds[0]
    assert x.shape == y.shape == (4,)

def test_evaluate_returns_measured_schema():
    model = AtlasModel(cfg())
    x = torch.randint(0, 256, (2, 8))
    loader = [(x, x)]
    metrics = evaluate(model, loader, max_batches=1)
    assert set(["loss", "perplexity", "batches", "seconds"]).issubset(metrics)
    assert metrics["batches"] == 1
