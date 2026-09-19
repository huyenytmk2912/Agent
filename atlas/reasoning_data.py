"""Deterministic reasoning curriculum generation for Atlas.

The generator creates structured text examples from the model-free reasoning
contract. It is deliberately synthetic and reproducible: no pretrained data,
weights, or paid services are required.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from atlas.reasoning import ReasoningEngine, ReasoningExample


@dataclass(frozen=True)
class CurriculumExample:
    prompt: str
    target: str
    difficulty: int


def render(example: ReasoningExample) -> CurriculumExample:
    engine = ReasoningEngine()
    predicted, trace = engine.solve(example)
    lines = [f"Question: {example.question}"]
    for step in trace:
        lines.append(f"{step.kind}: {step.value}")
    lines.append(f"Answer: {predicted}")
    return CurriculumExample(prompt=example.question, target="\n".join(lines), difficulty=len(example.operands))


def generate_examples(count: int = 32, seed: int = 7) -> list[CurriculumExample]:
    rng = random.Random(seed)
    examples: list[CurriculumExample] = []
    operations = ["add", "subtract", "multiply"]
    for _ in range(count):
        op = rng.choice(operations)
        width = rng.randint(2, 4)
        if op == "add":
            xs = tuple(rng.randint(-20, 20) for _ in range(width))
            question = "Compute the sum of " + ", ".join(map(str, xs)) + "."
        elif op == "subtract":
            xs = (rng.randint(0, 30),) + tuple(rng.randint(0, 10) for _ in range(width - 1))
            question = "Subtract " + " and ".join(map(str, xs[1:])) + f" from {xs[0]}."
        else:
            xs = tuple(rng.randint(-6, 6) for _ in range(width))
            question = "Compute the product of " + ", ".join(map(str, xs)) + "."
        answer = str(sum(xs) if op == "add" else (xs[0] - sum(xs[1:]) if op == "subtract" else _product(xs)))
        examples.append(render(ReasoningExample(question, answer, op, xs)))
    return examples


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def write_jsonl(path: str | Path, examples: Iterable[CurriculumExample]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
