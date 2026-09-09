from pathlib import Path
import torch

def save(path, model, optimizer, step, metrics=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'metrics':metrics or {}}, path)

def load(path, model, optimizer=None, map_location='cpu'):
    ck=torch.load(path,map_location=map_location)
    model.load_state_dict(ck['model'])
    if optimizer is not None: optimizer.load_state_dict(ck['optimizer'])
    return ck['step'], ck.get('metrics',{})
