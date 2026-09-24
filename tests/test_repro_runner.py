from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "limbo_repro_run", ROOT / "repro" / "run.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CONFIG_SPEC = importlib.util.spec_from_file_location(
    "limbo_config_loader", ROOT / "src" / "utils" / "config_loader.py"
)
assert CONFIG_SPEC is not None and CONFIG_SPEC.loader is not None
CONFIG_MODULE = importlib.util.module_from_spec(CONFIG_SPEC)
CONFIG_SPEC.loader.exec_module(CONFIG_MODULE)


class ReproManifestTest(unittest.TestCase):
    def test_all_configs_exist(self) -> None:
        experiments = MODULE.load_experiments()
        self.assertIn("qwen-os-limbo", experiments)
        for experiment in experiments.values():
            config = ROOT / experiment["config"]
            self.assertTrue(config.is_file(), config)
            loaded = CONFIG_MODULE.ConfigLoader().load_from(config)
            self.assertIn("assignment_config", loaded, config)

    def test_limbo_command_has_expected_controller_flags(self) -> None:
        experiment = MODULE.load_experiments()["qwen-os-limbo"]
        command = MODULE.build_command(experiment, seed=42, python="python")
        self.assertIn("--enable_general_bandit", command)
        self.assertIn("--bandit_retrieval_gated", command)
        self.assertIn("--bandit_budget_target", command)
        self.assertIn("0.000688", command)


if __name__ == "__main__":
    unittest.main()
