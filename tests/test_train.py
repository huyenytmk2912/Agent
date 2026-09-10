import torch
from atlas.model import AtlasModel
from atlas.train import train_step


def cfg():
    return type("Cfg", (), dict(vocab_size=64, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, ffn_dim=88, max_seq_len=16, rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True))


def test_train_step_returns_finite_loss_and_updates_weights():
    torch.manual_seed(7)
    model = AtlasModel(cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.tok.weight.detach().clone()
    x = torch.randint(0, 64, (2, 16)); y = x.clone()
    loss = train_step(model, opt, x, y)
    assert loss == loss and loss < 20
    assert not torch.equal(before, model.tok.weight.detach())
