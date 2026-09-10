"""Minimal CPU/GPU trainer for the Atlas random-initialized model."""
import argparse, math, random
from pathlib import Path
import torch
from .model import AtlasModel


def train_step(model, optimizer, x, y):
    model.train(); optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1))
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    return float(loss.detach())


def save_checkpoint(path, model, optimizer, step, metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "metrics": metrics}, path)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=10); p.add_argument("--vocab", type=int, default=256); p.add_argument("--seq", type=int, default=32); p.add_argument("--batch", type=int, default=2); p.add_argument("--out", default="runs/tiny/step.pt"); args = p.parse_args()
    random.seed(1337); torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = type("Cfg", (), dict(vocab_size=args.vocab, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, ffn_dim=352, max_seq_len=args.seq, rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True))
    model = AtlasModel(cfg).to(device); opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(1, args.steps + 1):
        x = torch.randint(0, args.vocab, (args.batch, args.seq), device=device); y = x.clone()
        loss = train_step(model, opt, x, y)
        if step == 1 or step == args.steps: print({"step": step, "loss": loss, "ppl": math.exp(min(loss, 20.0))})
    save_checkpoint(args.out, model, opt, args.steps, {"loss": loss, "device": device})

if __name__ == "__main__": main()
