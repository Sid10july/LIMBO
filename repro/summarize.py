#!/usr/bin/env python3
"""Extract top-line metrics and aggregate multiple LIMBO output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any


def nested_get(value: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = value
        try:
            for key in path:
                current = current[key]
        except (KeyError, TypeError):
            continue
        return current
    return None


def load_run(run_dir: Path) -> dict[str, Any]:
    metric_path = run_dir / "metric.json"
    if not metric_path.is_file():
        raise FileNotFoundError(f"Missing {metric_path}")
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    accuracy = nested_get(
        metric,
        ("overall", "evaluation_outcome", "correct"),
        ("evaluation_outcome", "correct"),
    )
    cost = nested_get(metric, ("cost", "total_cost_usd"), ("total_cost_usd",))
    tokens = nested_get(metric, ("cost", "total_tokens"), ("total_tokens",))
    if accuracy is None or cost is None:
        raise ValueError(f"Could not locate accuracy/cost in {metric_path}")
    return {
        "run": str(run_dir),
        "accuracy": float(accuracy),
        "total_cost_usd": float(cost),
        "total_tokens": int(tokens) if tokens is not None else None,
    }


def mean_sd(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    runs = [load_run(run_dir.resolve()) for run_dir in args.run_dirs]
    for run in runs:
        tokens = "n/a" if run["total_tokens"] is None else run["total_tokens"]
        print(
            f"{run['run']}: accuracy={run['accuracy']:.6f}, "
            f"cost=${run['total_cost_usd']:.8f}, tokens={tokens}"
        )

    accuracy_mean, accuracy_sd = mean_sd([run["accuracy"] for run in runs])
    cost_mean, cost_sd = mean_sd([run["total_cost_usd"] for run in runs])
    print(f"n={len(runs)}")
    print(f"accuracy_mean={accuracy_mean:.6f}")
    print(f"accuracy_sample_sd={accuracy_sd:.6f}")
    print(f"total_cost_mean_usd={cost_mean:.8f}")
    print(f"total_cost_sample_sd_usd={cost_sd:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
