#!/usr/bin/env python3
"""List, inspect, and launch named LIMBO experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "repro" / "experiments.yaml"


def load_experiments() -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    experiments = raw.get("experiments", {})
    if not isinstance(experiments, dict):
        raise ValueError("repro/experiments.yaml must contain an experiments mapping")
    return experiments


def build_command(experiment: dict[str, Any], seed: int, python: str) -> list[str]:
    config = ROOT / str(experiment["config"])
    if not config.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config}")
    extra_args = [str(value) for value in experiment.get("args", [])]
    return [
        python,
        str(ROOT / "src" / "run_experiment.py"),
        "--config_path",
        str(config),
        "--seed",
        str(seed),
        *extra_args,
    ]


def list_experiments(experiments: dict[str, dict[str, Any]]) -> None:
    width = max(len(name) for name in experiments)
    for name, experiment in experiments.items():
        description = experiment.get("description", "")
        print(f"{name:<{width}}  {description}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", help="experiment ID from the manifest")
    parser.add_argument("--list", action="store_true", help="list named experiments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="value for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter")
    parser.add_argument(
        "--dry-run", action="store_true", help="print without executing"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiments = load_experiments()
    if args.list:
        list_experiments(experiments)
        return 0
    if not args.experiment:
        raise SystemExit("Provide an experiment ID or use --list.")
    if args.experiment not in experiments:
        choices = ", ".join(sorted(experiments))
        raise SystemExit(f"Unknown experiment {args.experiment!r}. Choices: {choices}")

    experiment = experiments[args.experiment]
    required_env = experiment.get("requires_env")
    if required_env and not os.environ.get(str(required_env)) and not args.dry_run:
        raise SystemExit(f"Missing required environment variable: {required_env}")

    command = build_command(experiment, args.seed, args.python)
    print(f"experiment: {args.experiment}")
    print("command:")
    print("  " + shlex.join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.device is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.device
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
