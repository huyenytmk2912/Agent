"""Measured evaluation utilities; no fabricated benchmark values."""
import math
import time
import torch

def evaluate(model, loader, device="cpu", max_batches=None):
    model.eval(); losses = []; start = time.perf_counter()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches: break
            _, loss = model(x.to(device), y.to(device))
            losses.append(float(loss))
    mean = sum(losses) / max(1, len(losses))
    return {"loss": mean, "perplexity": math.exp(min(mean, 20.0)), "batches": len(losses), "seconds": time.perf_counter() - start}
