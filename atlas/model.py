"""Minimal decoder-only Transformer with random initialization only."""
import math
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)+self.eps).to(x.dtype) * self.weight

class MLP(nn.Module):
    def __init__(self, d_model, ffn_dim):
        super().__init__(); self.g=nn.Linear(d_model,ffn_dim,bias=False); self.u=nn.Linear(d_model,ffn_dim,bias=False); self.d=nn.Linear(ffn_dim,d_model,bias=False)
    def forward(self,x): return self.d(torch.nn.functional.silu(self.g(x))*self.u(x))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, eps=1e-5):
        super().__init__(); self.n1=RMSNorm(d_model,eps); self.attn=nn.MultiheadAttention(d_model,n_heads,batch_first=True); self.n2=RMSNorm(d_model,eps); self.mlp=MLP(d_model,ffn_dim)
    def forward(self,x):
        t=x.size(1); mask=torch.full((t,t),float('-inf'),device=x.device).triu(1)
        y,_=self.attn(self.n1(x),self.n1(x),self.n1(x),attn_mask=mask,need_weights=False)
        x=x+y; return x+self.mlp(self.n2(x))

class AtlasModel(nn.Module):
    def __init__(self,vocab_size=512,d_model=256,n_layers=4,n_heads=8,ffn_dim=704,tie_embeddings=True):
        super().__init__(); self.tok=nn.Embedding(vocab_size,d_model); self.blocks=nn.ModuleList([Block(d_model,n_heads,ffn_dim) for _ in range(n_layers)]); self.norm=RMSNorm(d_model); self.lm=nn.Linear(d_model,vocab_size,bias=False)
        if tie_embeddings: self.lm.weight=self.tok.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight,0.0,0.02)
    def forward(self,input_ids,labels=None):
        x=self.tok(input_ids)
        for b in self.blocks: x=b(x)
        logits=self.lm(self.norm(x)); loss=None
        if labels is not None: loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),labels[:,1:].reshape(-1))
        return logits,loss
