"""Deterministic reasoning scaffolding for Atlas.

This module is intentionally model-free: it provides a trace format, a small
verifier, and a benchmark harness that can later be connected to AtlasModel
without changing the evaluation contract.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ReasoningExample:
    question: str
    answer: str
    operation: str
    operands: tuple[int, ...]


@dataclass(frozen=True)
class TraceStep:
    kind: str
    value: str


class ReasoningEngine:
    """Small auditable executor for arithmetic reasoning examples."""

    def solve(self, example: ReasoningExample) -> tuple[str, list[TraceStep]]:
        op = example.operation
        xs = example.operands
        if not xs:
            raise ValueError("operands must not be empty")
        trace = [TraceStep("parse", f"operation={op}; operands={list(xs)}")]
        if op == "add":
            result = sum(xs)
        elif op == "subtract":
            result = xs[0]
            for value in xs[1:]:
                result -= value
        elif op == "multiply":
            result = 1
            for value in xs:
                result *= value
        elif op == "divide":
            result = xs[0]
            for value in xs[1:]:
                if value == 0:
                    raise ZeroDivisionError("division by zero")
                if result % value:
                    raise ValueError("non-integer intermediate result")
                result //= value
        else:
            raise ValueError(f"unsupported operation: {op}")
        trace.append(TraceStep("compute", str(result)))
        trace.append(TraceStep("verify", f"predicted={result}; expected={example.answer}"))
        return str(result), trace

    @staticmethod
    def validate_trace(example: ReasoningExample, predicted: str, trace: list[TraceStep]) -> bool:
        """Check that a trace has the required ordered steps and consistent values."""
        if [step.kind for step in trace] != ["parse", "compute", "verify"]:
            return False
        if f"operation={example.operation}" not in trace[0].value:
            return False
        if f"operands={list(example.operands)}" not in trace[0].value:
            return False
        if trace[1].value != predicted:
            return False
        return trace[2].value == f"predicted={predicted}; expected={example.answer}"

    def verify(self, example: ReasoningExample) -> dict[str, Any]:
        started = time.perf_counter()
        predicted, trace = self.solve(example)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        trace_valid = self.validate_trace(example, predicted, trace)
        return {
            "question": example.question,
            "expected": example.answer,
            "predicted": predicted,
            "correct": predicted == example.answer,
            "trace_valid": trace_valid,
            "latency_ms": elapsed_ms,
            "trace": [asdict(step) for step in trace],
        }


def benchmark(examples: Iterable[ReasoningExample]) -> dict[str, Any]:
    engine = ReasoningEngine()
    rows = [engine.verify(item) for item in examples]
    correct = sum(int(row["correct"]) for row in rows)
    trace_valid = sum(int(row["trace_valid"]) for row in rows)
    total = len(rows)
    latencies = [row["latency_ms"] for row in rows]
    return {
        "examples": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "trace_valid": trace_valid,
        "trace_valid_rate": (trace_valid / total) if total else 0.0,
        "mean_latency_ms": (sum(latencies) / total) if total else 0.0,
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "results": rows,
    }


def load_jsonl(path: str) -> list[ReasoningExample]:
    examples: list[ReasoningExample] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            examples.append(
                ReasoningExample(
                    question=item["question"],
                    answer=str(item["answer"]),
                    operation=item["operation"],
                    operands=tuple(int(x) for x in item["operands"]),
                )
            )
    return examples
