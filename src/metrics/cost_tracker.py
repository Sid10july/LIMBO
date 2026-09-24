# src/metrics/cost_tracker.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import json, os


@dataclass
class CallUsage:
    model: str
    prompt_tokens: int
    completion_tokens: int
    input_cost_usd: float
    output_cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


class CostRegistry:
    """
    Prices are in USD per 1K tokens.

    For hosted/API models, use the true provider prices. For local open-weight
    models, we use a standardized *notional* token schedule so runs remain
    comparable in paper tables instead of collapsing to $0.0.
    """

    STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE = {"in": 0.00004, "out": 0.00010}

    DEFAULTS: Dict[str, Dict[str, float]] = {
        # Qwen 2.5 7B Instruct (USD per 1K tokens)
        # Equivalent to $0.040 / 1M input and $0.100 / 1M output.
        "Qwen/Qwen2.5-7B-Instruct": {"in": 0.00004, "out": 0.00010},
        "qwen-2.5-7b-instruct": {"in": 0.00004, "out": 0.00010},
        # Matches the repo's existing GPT-4o mini estimate in
        # src/factories/data/standard_v0303/utility.py.
        "gpt-4o-mini": {"in": 0.00015, "out": 0.00060},
        "openai/gpt-4o-mini": {"in": 0.00015, "out": 0.00060},
        "meta-llama/Llama-3.1-8B-Instruct": STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
        "llama-3.1-8b-instruct": STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
        "DeepSeek-R1-Distill-Llama-8B": STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
        "deepseek-r1-distill-llama-8b": STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
        "DeepSeek-R1-Distill-Qwen-7B": STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
        "deepseek-r1-distill-qwen-7b": STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
    }

    PATTERN_DEFAULTS: tuple[tuple[tuple[str, ...], Dict[str, float]], ...] = (
        (
            (
                "qwen2.5-7b-instruct",
                "qwen/qwen2.5-7b-instruct",
                "/qwen2.5-7b-instruct",
            ),
            STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
        ),
        (
            (
                "llama-3.1-8b-instruct",
                "meta-llama/llama-3.1-8b-instruct",
                "/llama-3.1-8b-instruct",
            ),
            STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
        ),
        (
            (
                "deepseek-r1-distill-llama-8b",
                "/deepseek-r1-distill-llama-8b",
            ),
            STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
        ),
        (
            (
                "deepseek-r1-distill-qwen-7b",
                "/deepseek-r1-distill-qwen-7b",
            ),
            STANDARD_LOCAL_7B_8B_OPEN_MODEL_PRICE,
        ),
    )

    def __init__(self, overrides: Optional[Dict[str, Dict[str, float]]] = None):
        self._prices = dict(self.DEFAULTS)
        if overrides:
            self._prices.update(overrides)

    def get(self, model_name: str) -> Dict[str, float]:
        if model_name in self._prices:
            return self._prices[model_name]
        lowered_name = str(model_name).lower()
        for pattern_group, prices in self.PATTERN_DEFAULTS:
            if any(pattern in lowered_name for pattern in pattern_group):
                return dict(prices)
        return {"in": 0.0, "out": 0.0}


class CostTracker:
    def __init__(self, registry: Optional[CostRegistry] = None):
        self.registry = registry or CostRegistry()
        self.calls: list[CallUsage] = []

    def log_call(self, model_name: str, prompt_tokens: int, completion_tokens: int):
        prices = self.registry.get(model_name)
        input_cost = (prompt_tokens / 1000.0) * float(prices["in"])
        output_cost = (completion_tokens / 1000.0) * float(prices["out"])
        self.calls.append(
            CallUsage(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_cost_usd=input_cost,
                output_cost_usd=output_cost,
            )
        )

    def summary(self) -> Dict:
        total_prompt = sum(c.prompt_tokens for c in self.calls)
        total_completion = sum(c.completion_tokens for c in self.calls)
        per_model: Dict[str, Dict[str, float | int]] = {}
        for c in self.calls:
            d = per_model.setdefault(
                c.model,
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "input_cost_usd": 0.0,
                    "output_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                },
            )
            d["prompt_tokens"] += c.prompt_tokens
            d["completion_tokens"] += c.completion_tokens
            d["total_tokens"] += c.total_tokens
            d["input_cost_usd"] += c.input_cost_usd
            d["output_cost_usd"] += c.output_cost_usd
            d["total_cost_usd"] += c.total_cost_usd

        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": sum(c.total_cost_usd for c in self.calls),
            "per_model": per_model,
            "calls": [asdict(c) for c in self.calls],
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
