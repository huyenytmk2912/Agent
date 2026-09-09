"""Deterministic byte-level tokenizer; no pretrained vocabulary."""
class ByteTokenizer:
    def __init__(self): self.bos=256; self.eos=257; self.pad=258; self.unk=259
    @property
    def vocab_size(self): return 260
    def encode(self,text): return [self.bos]+list(text.encode('utf-8'))+[self.eos]
    def decode(self,ids): return bytes(i for i in ids if 0<=i<256).decode('utf-8',errors='replace')
