import torch
from atlas.model import AtlasModel
from atlas.tokenizer import ByteTokenizer

def test_random_model_forward():
    m=AtlasModel(vocab_size=260,d_model=64,n_layers=2,n_heads=4,ffn_dim=176)
    x=torch.randint(0,260,(2,8)); logits,loss=m(x,x)
    assert logits.shape==(2,8,260)
    assert torch.isfinite(loss)

def test_tokenizer_roundtrip():
    t=ByteTokenizer(); s='Atlas-7B from scratch'
    assert t.decode(t.encode(s))==s

def test_random_initialization():
    a=AtlasModel(vocab_size=260,d_model=32,n_layers=1,n_heads=4,ffn_dim=88)
    b=AtlasModel(vocab_size=260,d_model=32,n_layers=1,n_heads=4,ffn_dim=88)
    assert not torch.equal(a.tok.weight,b.tok.weight)
