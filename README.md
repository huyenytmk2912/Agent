# Atlas-7B: from-scratch AI foundation

A from-scratch language-model and coding-agent research stack. The brain is initialized from random weights and never loaded from a pretrained checkpoint.

## Target
- 7B decoder-only Transformer
- 4096 hidden, 32 layers, 32 query heads, 8 KV heads, 11008 SwiGLU intermediate
- 49152 vocabulary target
- CPU tiny fallback; GPU/DDP/FSDP-ready design

## Stages
Foundation -> tokenizer/data -> tiny training -> benchmarks -> reasoning/coding -> tools -> memory -> self-improvement -> coding agent/quant -> 7B scaling.

Training is not claimed complete until real checkpoints and metrics exist.
