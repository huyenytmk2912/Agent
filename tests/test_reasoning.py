import pytest

from atlas.reasoning import ReasoningEngine, benchmark, load_jsonl


def test_reasoning_trace_is_auditable():
    item = load_jsonl("tests/fixtures/reasoning_cases.jsonl")[0]
    predicted, trace = ReasoningEngine().solve(item)
    assert predicted == "12"
    assert [step.kind for step in trace] == ["parse", "compute", "verify"]


def test_reasoning_benchmark_reports_accuracy():
    report = benchmark(load_jsonl("tests/fixtures/reasoning_cases.jsonl"))
    assert report["examples"] == 4
    assert report["correct"] == 4
    assert report["accuracy"] == 1.0
    assert report["mean_latency_ms"] >= 0.0


def test_reasoning_rejects_invalid_division():
    with pytest.raises(ValueError):
        ReasoningEngine().solve(
            load_jsonl("tests/fixtures/reasoning_cases.jsonl")[0].__class__(
                question="bad", answer="", operation="divide", operands=(5, 2)
            )
        )
