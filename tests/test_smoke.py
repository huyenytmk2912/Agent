import json
from pathlib import Path

from atlas.smoke import run


def test_tiny_smoke_creates_metrics_and_checkpoint(tmp_path):
    corpus = Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl"
    metrics = run(str(corpus), str(tmp_path), steps=3, seq_len=16, batch_size=1)
    assert metrics["steps"] == 3
    assert metrics["batch_count"] > 0
    assert metrics["loss"] == metrics["loss"]
    assert metrics["last_train_loss"] == metrics["last_train_loss"]
    assert Path(tmp_path, "checkpoint.pt").exists()
    saved = json.loads(Path(tmp_path, "metrics.json").read_text())
    assert saved["steps"] == 3
