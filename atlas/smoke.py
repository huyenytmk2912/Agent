"""Deterministic tiny-corpus train/eval smoke test for Atlas.

This is a verification harness, not a quality claim: it proves the random-init
model, JSONL loader, trainer, evaluator, and checkpoint path work together.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch

from .data import JsonlDataset
from .evaluate import evaluate
from .model import AtlasModel
from .tokenizer import ByteTokenizer
from .train import save_checkpoint, train_step


def build_cfg(vocab_size: int, seq_len: int):
    return type(
        "Cfg",
        (),
        dict(
            vocab_size=vocab_size,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            ffn_dim=176,
            max_seq_len=seq_len,
            rope_theta=10000.0,
            norm_eps=1e-5,
            tie_embeddings=True,
        ),
    )


def run(corpus: str, out_dir: str, steps: int = 12, seq_len: int = 32, batch_size: int = 2):
    random.seed(1337)
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = ByteTokenizer()
    dataset = JsonlDataset(corpus, tok, seq_len=seq_len)
    if len(dataset) == 0:
        raise ValueError("corpus produced zero training examples")
    cfg = build_cfg(tok.vocab_size, seq_len)
    model = AtlasModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first_loss = None
    last_loss = None
    for step in range(1, steps + 1):
        indices = [step % len(dataset) for _ in range(batch_size)]
        x = torch.stack([dataset[i][0] for i in indices]).to(device)
        y = torch.stack([dataset[i][1] for i in indices]).to(device)
        last_loss = train_step(model, opt, x, y)
        if first_loss is None:
            first_loss = last_loss
    metrics = evaluate(model, dataset, batch_size=batch_size, device=device)
    metrics.update({"first_train_loss": first_loss, "last_train_loss": last_loss, "device": device, "steps": steps})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_checkpoint(out / "checkpoint.pt", model, opt, steps, metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out-dir", default="runs/tiny")
    p.add_argument("--steps", type=int, default=12)
    args = p.parse_args()
    metrics = run(args.corpus, args.out_dir, steps=args.steps)
    print(json.dumps(metrics, sort_keys=True))
    if not math.isfinite(metrics["loss"]):
        raise SystemExit("non-finite evaluation loss")


if __name__ == "__main__":
    main()
