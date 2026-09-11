from atlas.data import JsonlDataset
from atlas.tokenizer import ByteTokenizer

def test_jsonl_dataset(tmp_path):
    p=tmp_path/'data.jsonl'; p.write_text('{"text":"hello world"}\n', encoding='utf-8')
    ds=JsonlDataset(p, 4, ByteTokenizer())
    assert len(ds) >= 1
    x,y=ds[0]
    assert x.shape == y.shape
