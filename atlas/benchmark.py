"""Reproducible smoke benchmark; reports only measured values."""
import json, math, time
from pathlib import Path
import torch
from .data import JsonlDataset

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

def write_report(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

def benchmark(model, loader, device):
    model.eval(); losses=[]; start=time.time()
    with torch.no_grad():
        for x,y in loader:
            logits=model(x.to(device))
            loss=torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].to(device).reshape(-1))
            losses.append(float(loss))
    mean=sum(losses)/max(1,len(losses))
    return {"loss":mean,"perplexity":math.exp(min(mean,20.0)),"batches":len(losses),"seconds":time.time()-start,"parameters":parameter_count(model)}
