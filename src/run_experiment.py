import argparse
from collections import Counter
import json
import os
import yaml
import copy
import random
from enum import StrEnum
from typing import Any, Mapping, Sequence, Optional
import coredumpy  # type: ignore[import-untyped]

from src.utils import ConfigLoader, SingletonLogger, set_generation_token_budget
from src.typings import (
    AssignmentConfig,
    EnvironmentConfig,
    SampleStatus,
    LoggerConfig,
    ContinualAgentBenchException,
    Session,
    SessionEvaluationOutcome,
    SampleIndex,
    PathConfig,
    GeneralInstanceFactory,
    SessionMetricCalculationPartial,
    Role,
)
from src.tasks import Task, DatasetItem
from src.agents import Agent
from src.language_models import LanguageModel
from src.callbacks import (
    CallbackHandler,
    CallbackConstructor,
    Callback,
    CallbackRestorer,
    CallbackArguments,
)


class ConfigUtilityCaller(StrEnum):
    CLIENT = "client"
    SERVER = "server"
    CLIENT_SIDE_CONTROLLER = "client_side_controller"


class ConfigUtility:
    def __init__(
        self,
        assignment_config: AssignmentConfig,
        environment_config: EnvironmentConfig,
        path_config: PathConfig,
    ):
        self.assignment_config = assignment_config
        self.environment_config = environment_config
        self.path_config = path_config

    def preprocess(self) -> None:
        if self.environment_config.task_client:
            self.assignment_config.task = self.environment_config.task_client

    def construct(self) -> tuple[Task[DatasetItem], Agent, dict[str, Callback]]:
        # Maybe task will be Task or TaskClient, but it doesn't matter!
        task: Task[DatasetItem] = self.assignment_config.task.create()
        # region Construct language_model_dict
        # Here, We actually instantiate the language models.
        # After the code exit construct(), We will never get a chance to get the language model instance.
        # But I think this is good, since this improve the modularity and maintainability of the code.
        language_model_dict: Mapping[str, LanguageModel] = {
            key: value.create()
            for key, value in self.assignment_config.language_model_dict.items()
        }
        agent_instance_factory: GeneralInstanceFactory = self.assignment_config.agent
        if (
            language_model_name := agent_instance_factory.parameters.get(
                "language_model"
            )
        ) is not None:
            agent_instance_factory.parameters["language_model"] = language_model_dict[
                language_model_name
            ]
        # endregion
        agent: Agent = agent_instance_factory.create()
        callback_dict = CallbackConstructor.construct(
            self.assignment_config, task, agent, language_model_dict
        )
        return task, agent, callback_dict

    def validate(self, task: Task[DatasetItem], agent: Agent) -> None:
        sample_index_list = task.get_sample_index_list()
        if self.assignment_config.sample_order == "default":
            return
        # Normalize configured indices to the exact key type used by the task dataset.
        sample_index_set = set(sample_index_list)
        normalized_sample_order = []
        for selected_sample_index in self.assignment_config.sample_order:
            if selected_sample_index in sample_index_set:
                normalized_sample_order.append(selected_sample_index)
                continue
            selected_as_str = str(selected_sample_index)
            if selected_as_str in sample_index_set:
                normalized_sample_order.append(selected_as_str)
                continue
            try:
                selected_as_int = int(selected_sample_index)
            except (TypeError, ValueError):
                selected_as_int = None
            if selected_as_int is not None and selected_as_int in sample_index_set:
                normalized_sample_order.append(selected_as_int)
                continue
            assert selected_sample_index in sample_index_set
        self.assignment_config.sample_order = normalized_sample_order

    def postprocess(self, task: Task[DatasetItem], agent: Agent) -> None:
        if self.assignment_config.sample_order == "default":
            self.assignment_config.sample_order = task.get_sample_index_list()

    def remove_redundant_args(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        # Maybe use `if raw_config["environment_config"]["use_task_client_flag"]` is better, but I use the following
        # condition avoid using dict key directly.
        if not self.environment_config.task_client:
            # If the config file is used to restore the previous incomplete assignment, the `task_client` will be None.
            # Using del to remove the key-value pair will cause an error in this case.
            for key in list(raw_config["environment_config"]):
                if key != "use_task_client_flag":
                    del raw_config["environment_config"][key]
        redundant_key_buffer: set[tuple[str, str]] = set()
        for key in raw_config["task_dict"]:
            if key != raw_config["assignment_config"]["task"]:
                redundant_key_buffer.add(("task_dict", key))
        for key in raw_config["agent_dict"]:
            if key != raw_config["assignment_config"]["agent"]["name"]:
                redundant_key_buffer.add(("agent_dict", key))
        assignment_language_model_name_list: Sequence[str] = [
            language_model_info_dict["name"]
            for language_model_info_dict in raw_config["assignment_config"][
                "language_model_list"
            ]
        ]
        for key in raw_config["language_model_dict"]:
            if key not in assignment_language_model_name_list:
                redundant_key_buffer.add(("language_model_dict", key))
        assignment_callback_name_list: Sequence[str] = [
            callback_info_dict["name"]
            for callback_info_dict in raw_config["assignment_config"][
                "callback_dict"
            ].values()
        ]
        for key in raw_config["callback_dict"]:
            if key not in assignment_callback_name_list:
                redundant_key_buffer.add(("callback_dict", key))
        for info_tuple in redundant_key_buffer:
            del raw_config[info_tuple[0]][info_tuple[1]]
        return raw_config

    @staticmethod
    def _get_custom_instance_info_dict(
        default_instance_info_dict: Mapping[str, Any],
        custom_instance_info_dict: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        module: str = default_instance_info_dict["module"]
        # `... or {}` is used to ensure that both default_parameters and custom_parameters are dict.
        default_parameters: dict[str, Any] = copy.deepcopy(
            default_instance_info_dict.get("parameters") or {}
        )
        custom_parameters = custom_instance_info_dict.get("custom_parameters") or {}
        for parameter_name, custom_parameter_value in custom_parameters.items():
            assert parameter_name in default_parameters  # Do not remove this assertion.
            # Overwrite the default parameter value with the custom parameter value.
            default_parameters[parameter_name] = custom_parameter_value
        return {
            "module": module,
            "parameters": default_parameters,
        }

    @staticmethod
    def read_raw_config(
        raw_config: Mapping[str, Any], caller: ConfigUtilityCaller
    ) -> tuple[AssignmentConfig, EnvironmentConfig, LoggerConfig, PathConfig]:
        raw_config = copy.deepcopy(raw_config)  # Avoid modifying the original config.
        # region Convert raw_config into assignment_config
        # region Construct assignment_language_model_dict
        assignment_language_model_list: Sequence[Mapping[str, Any]] = raw_config[
            "assignment_config"
        ]["language_model_list"]
        assignment_language_model_dict: dict[str, Any] = {}
        for language_model_info_dict in assignment_language_model_list:
            language_model_name = language_model_info_dict["name"]
            default_language_model_info_dict = raw_config["language_model_dict"][
                language_model_name
            ]
            assert language_model_name not in assignment_language_model_dict
            assignment_language_model_dict[language_model_name] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_language_model_info_dict, language_model_info_dict
                )
            )
        # endregion
        # region Construct assignment_agent
        custom_agent_info_dict = raw_config["assignment_config"]["agent"]
        default_agent_info_dict = raw_config["agent_dict"][
            custom_agent_info_dict["name"]
        ]
        assignment_agent_info_dict = ConfigUtility._get_custom_instance_info_dict(
            default_agent_info_dict, custom_agent_info_dict
        )
        assignment_agent = GeneralInstanceFactory.model_validate(
            assignment_agent_info_dict
        )
        if (
            language_model_name := assignment_agent.parameters.get("language_model")
        ) is not None:
            # Do not replace the language_model in the parameters with the GeneralInstanceFactory instance.
            assert language_model_name in assignment_language_model_dict
        # endregion
        # region Construct assignment_callback_dict
        assignment_callback_dict: dict[str, Any] = raw_config["assignment_config"][
            "callback_dict"
        ]

        # DEBUG PRINTS — add these 4 lines:
        print(
            "Assignment callback names:",
            [cb["name"] for cb in assignment_callback_dict.values()],
        )
        print("Registry callback_dict keys:", list(raw_config["callback_dict"].keys()))
        # END DEBUG

        for callback_key, callback_info_dict in assignment_callback_dict.items():
            default_callback_info_dict = raw_config["callback_dict"][
                callback_info_dict["name"]
            ]
            assignment_callback_dict[callback_key] = (
                ConfigUtility._get_custom_instance_info_dict(
                    default_callback_info_dict, callback_info_dict
                )
            )
        # endregion
        assignment_config = AssignmentConfig(
            task=raw_config["task_dict"][raw_config["assignment_config"]["task"]],
            agent=assignment_agent,
            language_model_dict=assignment_language_model_dict,
            output_dir=raw_config["assignment_config"]["output_dir"],
            sample_order=raw_config["assignment_config"]["sample_order"],
            callback_dict=assignment_callback_dict,
        )
        # endregion
        # region Convert raw_config into environment_config
        if raw_config["environment_config"]["use_task_client_flag"]:
            environment_config = EnvironmentConfig(
                task_client=raw_config["environment_config"]["task_client"],
                chat_history_item_factory_client=raw_config["environment_config"][
                    "chat_history_item_factory_client"
                ],
                server_side_controller_address=raw_config["environment_config"][
                    "server_side_controller_address"
                ],
                interpreter_path=raw_config["environment_config"]["interpreter_path"],
            )
        else:
            environment_config = EnvironmentConfig(
                task_client=None,
                chat_history_item_factory_client=None,
                server_side_controller_address=None,
                interpreter_path=None,
            )
        # endregion
        # region Convert raw_config into logger_config
        if raw_config["logger_config"]["log_file_path"] == "default":
            if raw_config["environment_config"]["use_task_client_flag"]:
                match caller:
                    case ConfigUtilityCaller.CLIENT:
                        log_file_path = os.path.join(
                            assignment_config.output_dir, "singleton_logger_client.log"
                        )
                    case ConfigUtilityCaller.SERVER:
                        log_file_path = os.path.join(
                            assignment_config.output_dir, "singleton_logger_server.log"
                        )
                    case ConfigUtilityCaller.CLIENT_SIDE_CONTROLLER:
                        log_file_path = (
                            "./outputs/singleton_logger_client_side_controller.log"
                        )
                    case _:
                        raise NotImplementedError()
            else:
                log_file_path = os.path.join(
                    assignment_config.output_dir, "singleton_logger.log"
                )
        else:
            log_file_path = raw_config["logger_config"]["log_file_path"]
        logger_config = LoggerConfig(
            level=raw_config["logger_config"]["level"],
            log_file_path=log_file_path,
            logger_name=raw_config["logger_config"]["logger_name"],
        )
        # endregion
        # region Construct path_config from assignment_config
        path_config = PathConfig(
            exception_record_file_path=os.path.join(
                assignment_config.output_dir, "exception.txt"
            ),
            config_output_path=os.path.join(
                assignment_config.output_dir, "config.yaml"
            ),
            session_list_output_path=os.path.join(
                assignment_config.output_dir, "runs.json"
            ),
            metric_output_path=os.path.join(
                assignment_config.output_dir, "metric.json"
            ),
            coredumpy_output_dir=os.path.join(
                assignment_config.output_dir, "coredumpy"
            ),
        )
        # endregion
        return assignment_config, environment_config, logger_config, path_config

    @staticmethod
    def is_raw_config_equal(
        raw_config_1: dict[str, Any], raw_config_2: dict[str, Any]
    ) -> bool:
        raw_config_1 = copy.deepcopy(raw_config_1)
        raw_config_2 = copy.deepcopy(raw_config_2)
        output_dir_1 = raw_config_1["assignment_config"].pop("output_dir")
        output_dir_2 = raw_config_2["assignment_config"].pop("output_dir")
        return raw_config_1 == raw_config_2 and AssignmentConfig.is_output_dir_equal(
            output_dir_1, output_dir_2
        )


def main() -> None:
    # region Prepare variables
    os.environ.setdefault("CC", "/usr/bin/gcc")
    os.environ.setdefault("CXX", "/usr/bin/g++")
    os.environ.setdefault("TRITON_CC", "/usr/bin/gcc")
    os.environ.setdefault("TRITON_CXX", "/usr/bin/g++")
    os.environ.setdefault("CMAKE_C_COMPILER", "/usr/bin/gcc")
    os.environ.setdefault("CMAKE_CXX_COMPILER", "/usr/bin/g++")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("LAB_RUN_SEED", "42")),
    )
    parser.add_argument(
        "--enable_dbbench_bandit",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_ENABLE", "0") == "1",
    )
    parser.add_argument(
        "--enable_general_bandit",
        action="store_true",
        default=os.environ.get("GENERAL_BANDIT_ENABLE", "0") == "1",
    )
    parser.add_argument(
        "--enable_dbbench_art_lora",
        action="store_true",
        default=os.environ.get("DBBENCH_ART_ENABLE", "0") == "1",
    )
    parser.add_argument(
        "--dbbench_art_adapter_path",
        type=str,
        default=os.environ.get("DBBENCH_ART_ADAPTER_PATH", ""),
    )
    parser.add_argument(
        "--dbbench_art_base_model",
        type=str,
        default=os.environ.get("DBBENCH_ART_BASE_MODEL", ""),
    )
    parser.add_argument(
        "--dbbench_art_allocator_base_model",
        type=str,
        default=os.environ.get(
            "DBBENCH_ART_ALLOCATOR_BASE_MODEL",
            os.environ.get("DBBENCH_ART_BASE_MODEL", ""),
        ),
    )
    parser.add_argument(
        "--dbbench_art_reward_lambda",
        type=float,
        default=float(os.environ.get("DBBENCH_ART_REWARD_LAMBDA", "0.3")),
    )
    parser.add_argument(
        "--dbbench_art_budget_target",
        type=float,
        default=float(os.environ.get("DBBENCH_ART_BUDGET_TARGET", "0.00009")),
    )
    parser.add_argument(
        "--dbbench_art_action_set",
        type=str,
        default=os.environ.get("DBBENCH_ART_ACTION_SET", "tool_friendly_v1"),
    )
    parser.add_argument(
        "--dbbench_art_gpu_memory_utilization",
        type=float,
        default=float(os.environ.get("DBBENCH_ART_GPU_MEMORY_UTILIZATION", "0.25")),
    )
    parser.add_argument(
        "--dbbench_art_max_model_len",
        type=int,
        default=int(os.environ.get("DBBENCH_ART_MAX_MODEL_LEN", "2048")),
    )
    parser.add_argument(
        "--bandit_lambda",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_LAMBDA", "0.3")),
    )
    parser.add_argument(
        "--bandit_alpha",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_ALPHA", "0.5")),
    )
    parser.add_argument(
        "--bandit_budget_target",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_BUDGET_TARGET", "0.00009")),
    )
    parser.add_argument(
        "--bandit_lambda_lr",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_LAMBDA_LR", "0.05")),
    )
    parser.add_argument(
        "--bandit_adaptive_lambda",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_ADAPTIVE_LAMBDA", "1") == "1",
    )
    parser.add_argument(
        "--bandit_two_head",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_TWO_HEAD", "1") == "1",
    )
    parser.add_argument(
        "--bandit_two_timescale",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_TWO_TIMESCALE", "0") == "1",
    )
    parser.add_argument(
        "--bandit_correct_head_lr",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_CORRECT_HEAD_LR", "0.25")),
    )
    parser.add_argument(
        "--bandit_cost_head_lr",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_COST_HEAD_LR", "1.0")),
    )
    parser.add_argument(
        "--bandit_residual_updates",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESIDUAL_UPDATES", "0") == "1",
    )
    parser.add_argument(
        "--bandit_richer_features",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RICHER_FEATURES", "0") == "1",
    )
    parser.add_argument(
        "--bandit_feature_subset",
        type=str,
        default=os.environ.get("DBBENCH_BANDIT_FEATURE_SUBSET", ""),
        help=(
            "Comma-separated bandit feature names. "
            "If omitted, use the default generalizable feature set "
            "(plus richer DB-specific features when --bandit_richer_features is enabled)."
        ),
    )
    parser.add_argument(
        "--bandit_budget_only_baseline",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_BUDGET_ONLY_BASELINE", "0") == "1",
        help=(
            "Ablation baseline: keep the adaptive controller/action space but "
            "remove retrieval-conditioned features so the policy only sees "
            "prompt/task/budget signals."
        ),
    )
    parser.add_argument(
        "--bandit_decoupled_action_space",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_DECOUPLED_ACTION_SPACE", "0") == "1",
    )
    parser.add_argument(
        "--bandit_warm_start_slow_adaptation",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_WARM_START_SLOW_ADAPTATION", "0")
        == "1",
    )
    parser.add_argument(
        "--bandit_warm_start_samples",
        type=int,
        default=int(os.environ.get("DBBENCH_BANDIT_WARM_START_SAMPLES", "100")),
    )
    parser.add_argument(
        "--bandit_post_warmup_scale",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_POST_WARMUP_SCALE", "0.25")),
    )
    parser.add_argument(
        "--bandit_policy",
        type=str,
        default=os.environ.get("DBBENCH_BANDIT_POLICY", "utility"),
        choices=["cheapest_safe", "utility", "fixed_tau"],
    )
    parser.add_argument(
        "--bandit_correctness_threshold",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_CORRECTNESS_THRESHOLD", "0.70")),
    )
    parser.add_argument(
        "--bandit_adaptive_threshold",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_ADAPTIVE_THRESHOLD", "1") == "1",
    )
    parser.add_argument(
        "--bandit_target_accuracy",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_TARGET_ACCURACY", "0.68")),
    )
    parser.add_argument(
        "--bandit_accuracy_tolerance",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_ACCURACY_TOL", "0.01")),
    )
    parser.add_argument(
        "--bandit_threshold_step",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_THRESHOLD_STEP", "0.005")),
    )
    parser.add_argument(
        "--bandit_cost_ema_decay",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_COST_EMA_DECAY", "0.9")),
    )
    parser.add_argument(
        "--bandit_rescue_enable",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_ENABLE", "1") == "1",
    )
    parser.add_argument(
        "--bandit_rescue_profile_name",
        type=str,
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_PROFILE", "budget_512_r3"),
    )
    parser.add_argument(
        "--bandit_rescue_on_incorrect",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_ON_INCORRECT", "0") == "1",
    )
    parser.add_argument(
        "--bandit_rescue_on_task_limit",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_ON_TASK_LIMIT", "1") == "1",
    )
    parser.add_argument(
        "--bandit_rescue_on_incomplete",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_ON_INCOMPLETE", "0") == "1",
    )
    parser.add_argument(
        "--bandit_threshold_min",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_THRESHOLD_MIN", "0.55")),
    )
    parser.add_argument(
        "--bandit_threshold_max",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_THRESHOLD_MAX", "0.90")),
    )
    parser.add_argument(
        "--bandit_allow_max_arm",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_ALLOW_MAX_ARM", "0") == "1",
    )
    parser.add_argument(
        "--bandit_disallow_lowest_arm",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_DISALLOW_LOWEST_ARM", "0") == "1",
    )
    parser.add_argument(
        "--bandit_min_predicted_correct_floor",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_MIN_PREDICTED_CORRECT_FLOOR", "0.0")),
    )
    parser.add_argument(
        "--bandit_cost_guardrail_enable",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_COST_GUARDRAIL_ENABLE", "1") == "1",
    )
    parser.add_argument(
        "--bandit_cost_guardrail_band",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_COST_GUARDRAIL_BAND", "0.10")),
    )
    parser.add_argument(
        "--bandit_cost_guardrail_decay",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_COST_GUARDRAIL_DECAY", "0.95")),
    )
    parser.add_argument(
        "--bandit_replay_gated",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_REPLAY_GATED", "0") == "1",
        help=(
            "Use REPLAY_GATED_BUDGET_PRIMITIVES as the action space so the bandit "
            "jointly decides whether to replay (none/full/compressed) and which "
            "compute budget to use.  Requires a replay callback in the config; "
            "the callback reads task._copal_replay_mode before injecting."
        ),
    )
    parser.add_argument(
        "--bandit_retrieval_gated",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RETRIEVAL_GATED", "0") == "1",
        help=(
            "Use RETRIEVAL_GATED_BUDGET_PRIMITIVES as the action space so the "
            "bandit jointly decides whether to skip replay, use recent full replay, "
            "or use retrieved replay. Requires RetrievalPreviousSampleUtilizationCallback "
            "in the config."
        ),
    )
    parser.add_argument(
        "--bandit_retrieval_size_gated",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RETRIEVAL_SIZE_GATED", "0") == "1",
        help=(
            "Use RETRIEVAL_SIZE_GATED_BUDGET_PRIMITIVES as the action space so "
            "the bandit chooses among no replay and retrieved replay sizes "
            "(top-4/top-8/top-16) while holding compute budgets comparable."
        ),
    )
    parser.add_argument(
        "--bandit_replay_heavy_retrieval_size_gated",
        action="store_true",
        default=os.environ.get(
            "DBBENCH_BANDIT_REPLAY_HEAVY_RETRIEVAL_SIZE_GATED", "0"
        )
        == "1",
        help=(
            "Use a replay-heavy retrieval-size action space with one no-replay "
            "fallback and stronger retrieved-replay arms. Intended for "
            "replay-dominant regimes such as Llama DB."
        ),
    )
    parser.add_argument(
        "--heuristic_replay_gate_enable",
        action="store_true",
        default=os.environ.get("COPAL_HEURISTIC_REPLAY_GATE_ENABLE", "0") == "1",
        help=(
            "Baseline: replace the learned replay gate with a fixed heuristic "
            "based on retrieval similarity, skill count, and recent task-limit rate."
        ),
    )
    parser.add_argument(
        "--heuristic_replay_rule",
        type=str,
        default=os.environ.get("COPAL_HEURISTIC_REPLAY_RULE", "any"),
        choices=["any", "all"],
    )
    parser.add_argument(
        "--heuristic_replay_similarity_threshold",
        type=float,
        default=float(
            os.environ.get("COPAL_HEURISTIC_REPLAY_SIMILARITY_THRESHOLD", "0.55")
        ),
    )
    parser.add_argument(
        "--heuristic_replay_skill_threshold",
        type=int,
        default=int(os.environ.get("COPAL_HEURISTIC_REPLAY_SKILL_THRESHOLD", "4")),
    )
    parser.add_argument(
        "--heuristic_replay_task_limit_threshold",
        type=float,
        default=float(
            os.environ.get("COPAL_HEURISTIC_REPLAY_TASK_LIMIT_THRESHOLD", "0.35")
        ),
    )
    parser.add_argument(
        "--heuristic_replay_mode",
        type=str,
        default=os.environ.get("COPAL_HEURISTIC_REPLAY_MODE", "retrieved"),
        choices=["full", "retrieved", "compressed"],
    )
    parser.add_argument(
        "--heuristic_replay_sample_count",
        type=int,
        default=int(os.environ.get("COPAL_HEURISTIC_REPLAY_SAMPLE_COUNT", "1")),
    )
    parser.add_argument(
        "--self_consistency_enable",
        action="store_true",
        default=os.environ.get("COPAL_SELF_CONSISTENCY_ENABLE", "0") == "1",
        help=(
            "Baseline: run multiple stochastic attempts from the same initial state "
            "and pick the majority trace signature without using evaluation labels."
        ),
    )
    parser.add_argument(
        "--self_consistency_replay_mode",
        type=str,
        default=os.environ.get("COPAL_SELF_CONSISTENCY_REPLAY_MODE", "none"),
        choices=["none", "full", "retrieved", "compressed"],
        help=(
            "Replay mode for self-consistency attempts. Default 'none' gives the "
            "clean compute-on-attempts baseline instead of conflating it with replay."
        ),
    )
    parser.add_argument(
        "--self_consistency_replay_sample_count",
        type=int,
        default=int(os.environ.get("COPAL_SELF_CONSISTENCY_REPLAY_SAMPLE_COUNT", "1")),
        help=(
            "Replay count override for self-consistency when replay_mode='retrieved'. "
            "Ignored for replay_mode='none'."
        ),
    )
    parser.add_argument(
        "--self_consistency_max_attempts",
        type=int,
        default=int(os.environ.get("COPAL_SELF_CONSISTENCY_MAX_ATTEMPTS", "4")),
    )
    parser.add_argument(
        "--self_consistency_cost_budget_usd",
        type=float,
        default=float(os.environ.get("COPAL_SELF_CONSISTENCY_COST_BUDGET_USD", "0.0")),
        help=(
            "Optional soft cost cap for repeated attempts. "
            "When > 0, stop after exceeding this total per-sample attempt cost."
        ),
    )
    parser.add_argument(
        "--self_consistency_do_sample",
        action="store_true",
        default=os.environ.get("COPAL_SELF_CONSISTENCY_DO_SAMPLE", "1") == "1",
    )
    parser.add_argument(
        "--self_consistency_temperature",
        type=float,
        default=float(os.environ.get("COPAL_SELF_CONSISTENCY_TEMPERATURE", "0.7")),
    )
    parser.add_argument(
        "--self_consistency_top_p",
        type=float,
        default=float(os.environ.get("COPAL_SELF_CONSISTENCY_TOP_P", "0.95")),
    )
    parser.add_argument(
        "--enable_inference_rl",
        action="store_true",
        default=os.environ.get("COPAL_INFERENCE_RL_ENABLE", "0") == "1",
        help=(
            "OS Interaction controller: learn a sequential inference-time policy "
            "over retrieval, budget, reasoning, verification, retry, and termination."
        ),
    )
    parser.add_argument(
        "--rl_algorithm",
        type=str,
        default=os.environ.get("COPAL_RL_ALGORITHM", "masked_double_dqn"),
        choices=["masked_double_dqn"],
    )
    parser.add_argument(
        "--rl_max_decisions_per_sample",
        type=int,
        default=int(os.environ.get("COPAL_RL_MAX_DECISIONS_PER_SAMPLE", "12")),
    )
    parser.add_argument(
        "--rl_max_attempts",
        type=int,
        default=int(os.environ.get("COPAL_RL_MAX_ATTEMPTS", "2")),
    )
    parser.add_argument(
        "--rl_max_retrieval_injections",
        type=int,
        default=int(os.environ.get("COPAL_RL_MAX_RETRIEVAL_INJECTIONS", "2")),
    )
    parser.add_argument(
        "--rl_max_verifications",
        type=int,
        default=int(os.environ.get("COPAL_RL_MAX_VERIFICATIONS", "1")),
    )
    parser.add_argument(
        "--rl_budget_target",
        type=float,
        default=float(os.environ.get("COPAL_RL_BUDGET_TARGET", "0.000608")),
    )
    parser.add_argument(
        "--rl_cost_lambda",
        type=float,
        default=float(os.environ.get("COPAL_RL_COST_LAMBDA", "0.12")),
    )
    parser.add_argument(
        "--rl_gamma",
        type=float,
        default=float(os.environ.get("COPAL_RL_GAMMA", "0.95")),
    )
    parser.add_argument(
        "--rl_lr",
        type=float,
        default=float(os.environ.get("COPAL_RL_LR", "0.001")),
    )
    parser.add_argument(
        "--rl_epsilon_start",
        type=float,
        default=float(os.environ.get("COPAL_RL_EPSILON_START", "0.25")),
    )
    parser.add_argument(
        "--rl_epsilon_end",
        type=float,
        default=float(os.environ.get("COPAL_RL_EPSILON_END", "0.05")),
    )
    parser.add_argument(
        "--rl_target_update_interval",
        type=int,
        default=int(os.environ.get("COPAL_RL_TARGET_UPDATE_INTERVAL", "50")),
    )
    parser.add_argument(
        "--rl_replay_buffer_size",
        type=int,
        default=int(os.environ.get("COPAL_RL_REPLAY_BUFFER_SIZE", "4096")),
    )
    parser.add_argument(
        "--rl_verify_mode",
        type=str,
        default=os.environ.get("COPAL_RL_VERIFY_MODE", "self_critique"),
        choices=["none", "self_critique"],
    )
    parser.add_argument(
        "--rl_action_set",
        type=str,
        default=os.environ.get("COPAL_RL_ACTION_SET", "os_v1"),
        choices=["os_v1"],
    )
    parser.add_argument(
        "--bandit_rescue_confidence_gate",
        action="store_true",
        default=os.environ.get("DBBENCH_BANDIT_RESCUE_CONFIDENCE_GATE", "0") == "1",
    )
    parser.add_argument(
        "--bandit_rescue_predicted_correct_max",
        type=float,
        default=float(
            os.environ.get("DBBENCH_BANDIT_RESCUE_PREDICTED_CORRECT_MAX", "0.72")
        ),
    )
    parser.add_argument(
        "--bandit_rescue_uncertainty_min",
        type=float,
        default=float(os.environ.get("DBBENCH_BANDIT_RESCUE_UNCERTAINTY_MIN", "0.09")),
    )
    args = parser.parse_args()
    if sum(
        int(flag)
        for flag in (
            args.bandit_replay_gated,
            args.bandit_retrieval_gated,
            args.bandit_retrieval_size_gated,
            args.bandit_replay_heavy_retrieval_size_gated,
        )
    ) > 1:
        raise ValueError(
            "Choose at most one gated replay action space: "
            "--bandit_replay_gated, --bandit_retrieval_gated, or "
            "--bandit_retrieval_size_gated, or "
            "--bandit_replay_heavy_retrieval_size_gated."
        )
    if (args.enable_dbbench_bandit or args.enable_general_bandit) and args.enable_dbbench_art_lora:
        raise ValueError(
            "ART+LoRA and bandit routing are mutually exclusive. "
            "Enable only one controller path per run."
        )
    if args.self_consistency_enable and (
        args.enable_dbbench_bandit
        or args.enable_general_bandit
        or args.enable_dbbench_art_lora
        or args.enable_inference_rl
    ):
        raise ValueError(
            "Self-consistency baseline is an alternative controller. "
            "Do not combine it with ART, bandit routing, or inference RL."
        )
    if args.heuristic_replay_gate_enable and (
        args.enable_dbbench_bandit
        or args.enable_general_bandit
        or args.enable_inference_rl
    ):
        raise ValueError(
            "Heuristic replay gate is a non-bandit routing baseline. "
            "Do not combine it with bandit routing or inference RL."
        )
    if args.enable_inference_rl and (
        args.enable_dbbench_bandit
        or args.enable_general_bandit
        or args.enable_dbbench_art_lora
    ):
        raise ValueError(
            "Inference RL is an alternative controller. "
            "Enable only one of RL, ART, or bandit routing."
        )
    if args.self_consistency_max_attempts <= 0:
        raise ValueError("--self_consistency_max_attempts must be positive.")
    if args.self_consistency_replay_sample_count <= 0:
        raise ValueError("--self_consistency_replay_sample_count must be positive.")
    if args.heuristic_replay_sample_count <= 0:
        raise ValueError("--heuristic_replay_sample_count must be positive.")
    if args.rl_max_decisions_per_sample <= 0:
        raise ValueError("--rl_max_decisions_per_sample must be positive.")
    if args.rl_max_attempts <= 0:
        raise ValueError("--rl_max_attempts must be positive.")
    if args.rl_max_retrieval_injections < 0:
        raise ValueError("--rl_max_retrieval_injections must be non-negative.")
    if args.rl_max_verifications < 0:
        raise ValueError("--rl_max_verifications must be non-negative.")
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass
    raw_config = ConfigLoader().load_from(args.config_path)
    assignment_config, environment_config, logger_config, path_config = (
        ConfigUtility.read_raw_config(raw_config, ConfigUtilityCaller.CLIENT)
    )
    # Prepare the variable that will be used in the following procedures.
    config_utility = ConfigUtility(assignment_config, environment_config, path_config)
    # endregion
    # region write raw_config to disk
    cleaned_config = config_utility.remove_redundant_args(raw_config)
    config_output_path = path_config.config_output_path
    if os.path.exists(config_output_path):
        config_from_disk = yaml.safe_load(open(config_output_path, "r"))
        assert ConfigUtility.is_raw_config_equal(config_from_disk, cleaned_config)
        # The config file already exists, so we don't need to write it again.
    else:
        # Write the config file to the output directory.
        config_output_dir = os.path.dirname(config_output_path)
        if not os.path.exists(config_output_dir):
            os.makedirs(config_output_dir)
        yaml.dump(
            cleaned_config,
            open(config_output_path, "w"),
        )
    # endregion
    # region Initialize logger, Set coredumpy output dir
    logger = SingletonLogger.get_instance(logger_config)
    coredumpy.patch_except(directory=path_config.coredumpy_output_dir)
    # endregion
    # region Construct variable, valid config
    config_utility.preprocess()
    task, agent, callback_dict = config_utility.construct()
    #####
    from src.metrics.cost_tracker import CostTracker, CostRegistry
    from src.metrics.llm_counting_wrapper import CountingLLM

    # create a tracker for this run
    cost_tracker = CostTracker(
        # Optional: define prices here if you’re using a paid provider
        # CostRegistry({"meta-llama/Llama-3.1-8B-Instruct": {"in": 0.0, "out": 0.0}})
    )

    # Figure out model name + tokenizer on the agent/llm object
    llm_obj = getattr(agent, "llm", None) or getattr(agent, "model", None)
    model_name = None
    for attr in ("model_name_or_path", "model_id", "name", "model"):
        if hasattr(llm_obj, attr):
            model_name = getattr(llm_obj, attr)
            break
    model_name = str(model_name or "unknown-model")

    tokenizer = getattr(llm_obj, "tokenizer", None)

    # Wrap
    wrapped = CountingLLM(
        llm_obj, model_name=model_name, tokenizer=tokenizer, cost_tracker=cost_tracker
    )
    if hasattr(agent, "llm"):
        agent.llm = wrapped
    elif hasattr(agent, "model"):
        agent.model = wrapped
    else:
        # Worst case: the agent keeps a callable function attribute — try common names:
        setattr(agent, "llm", wrapped)
    # >>> ADD THIS so callbacks can find the tracker <<<
    setattr(agent, "cost_tracker", cost_tracker)
    # Also attach to language_model (used by LanguageModelAgent)
    lm_obj = getattr(agent, "_language_model", None)
    if lm_obj is not None:
        setattr(lm_obj, "cost_tracker", cost_tracker)
    #####
    config_utility.postprocess(task, agent)
    config_utility.validate(task, agent)
    ContinualAgentBenchException.set_record_file(path_config.exception_record_file_path)
    # endregion
    # region Determine whether to start a new assignment or restore the previous incomplete assignment, based on
    # whether the config file exists.
    session_list_output_path = path_config.session_list_output_path
    assert isinstance(assignment_config.sample_order, list)
    session_list: list[Session]
    unfinished_sample_order: list[SampleIndex]
    if os.path.exists(session_list_output_path):
        # At least one session exists, so we restore the previous incomplete assignment.
        session_list = [
            Session.model_validate(session_info_dict)
            for session_info_dict in json.load(open(session_list_output_path, "r"))
        ]
        unfinished_sample_order = [
            sample_index
            for sample_index in assignment_config.sample_order
            if all(session.sample_index != sample_index for session in session_list)
        ]
        # Previous session may change the state of the callback, restore it here.
        CallbackRestorer.restore(callback_dict)
    else:
        # Start a new assignment.
        session_list = []
        unfinished_sample_order = assignment_config.sample_order
    callback_handler = CallbackHandler(callback_dict)
    # endregion
    # region Run experiment
    task_name_str = str(task.task_name)
    bandit_enabled = bool(args.enable_dbbench_bandit or args.enable_general_bandit)
    art_enabled = bool(args.enable_dbbench_art_lora) and task_name_str == "db_bench"
    inference_rl_enabled = bool(args.enable_inference_rl)
    inference_rl_controller = None
    inference_rl_log_path = os.path.join(
        assignment_config.output_dir, "metrics", "inference_rl.jsonl"
    )
    inference_rl_state_path = os.path.join(
        assignment_config.output_dir, "metrics", "inference_rl_state.pt"
    )
    inference_rl_summary_path = os.path.join(
        assignment_config.output_dir, "metrics", "inference_rl_summary.json"
    )
    inference_rl_running_correct_count = 0
    inference_rl_running_sample_count = 0
    dbbench_bandit = None
    dbbench_feature_adapter = None
    dbbench_budget_primitives = []
    dbbench_art_policy = None
    dbbench_art_config = None
    dbbench_art_manifest: dict[str, Any] = {}
    dbbench_art_log_path = os.path.join(
        assignment_config.output_dir, "metrics", "dbbench_art_lora.jsonl"
    )
    dbbench_art_action_counts: dict[str, int] = {}
    art_state_by_sample: dict[str, dict[str, Any]] = {}
    bandit_state_by_sample: dict[str, dict[str, Any]] = {}
    bandit_action_counts: list[int] = []
    bandit_action_cost_ema_normalized: list[Optional[float]] = []
    bandit_action_cost_ema_usd: list[Optional[float]] = []
    rescue_action_counts: list[int] = []
    current_bandit_lambda = float(args.bandit_lambda)
    current_correctness_threshold = float(args.bandit_correctness_threshold)
    bandit_lambda_history: list[float] = []
    bandit_threshold_history: list[float] = []
    bandit_running_correct_count = 0
    bandit_running_sample_count = 0
    bandit_rescue_count = 0
    bandit_rescue_success_count = 0
    bandit_rescue_enabled = False
    previous_sample_cost_normalized = 0.0
    previous_sample_correct = False
    previous_sample_had_error = False
    previous_action_uncertainty = 0.0
    running_cost_per_sample_ema_usd: Optional[float] = None
    running_cost_per_sample_ema_normalized: Optional[float] = None
    running_cost_short_ema_normalized: Optional[float] = None
    running_cost_long_ema_normalized: Optional[float] = None
    recent_task_limit_rate = 0.0
    replay_reward_ema: Optional[float] = None
    no_replay_reward_ema: Optional[float] = None
    replay_advantage_ema = 0.0
    successful_task_token_sets: list[set[str]] = []
    threshold_min_hit_count = 0
    threshold_max_hit_count = 0
    non_completed_status_count: dict[str, int] = {}
    rescue_profile_idx = 0
    min_budget_profile_idx = 0
    max_budget_profile_idx = 0
    allowed_policy_action_indices: list[int] = []
    correct_head_lr_scale = (
        max(0.0, float(args.bandit_correct_head_lr))
        if args.bandit_two_timescale
        else 1.0
    )
    cost_head_lr_scale = (
        max(0.0, float(args.bandit_cost_head_lr))
        if args.bandit_two_timescale
        else 1.0
    )
    warm_start_samples = max(0, int(args.bandit_warm_start_samples))
    post_warmup_scale = min(1.0, max(0.0, float(args.bandit_post_warmup_scale)))
    bandit_log_path = os.path.join(
        assignment_config.output_dir, "metrics", "dbbench_bandit.jsonl"
    )
    heuristic_log_path = os.path.join(
        assignment_config.output_dir, "metrics", "heuristic_replay_gate.jsonl"
    )
    self_consistency_log_path = os.path.join(
        assignment_config.output_dir, "metrics", "self_consistency.jsonl"
    )
    bandit_budget_target = max(float(args.bandit_budget_target), 1e-9)
    if inference_rl_enabled:
        if task_name_str != "os_interaction":
            raise ValueError(
                "--enable_inference_rl currently supports only task=os_interaction."
            )
        from src.controllers.inference_allocation_rl import (
            MaskedDQNConfig,
            MaskedDQNInferenceController,
            OS_RL_ACTIONS,
            OS_RL_FEATURE_NAMES,
            run_os_inference_rl_episode,
        )

        inference_rl_config = MaskedDQNConfig(
            state_dim=len(OS_RL_FEATURE_NAMES),
            action_names=OS_RL_ACTIONS,
            gamma=float(args.rl_gamma),
            lr=float(args.rl_lr),
            epsilon_start=float(args.rl_epsilon_start),
            epsilon_end=float(args.rl_epsilon_end),
            target_update_interval=int(args.rl_target_update_interval),
            replay_buffer_size=int(args.rl_replay_buffer_size),
            seed=int(args.seed),
        )
        inference_rl_controller = MaskedDQNInferenceController(inference_rl_config)
        if os.path.exists(inference_rl_state_path):
            inference_rl_controller.load_state(inference_rl_state_path)
        if os.path.exists(inference_rl_log_path):
            try:
                restored_rows_by_sample: dict[str, dict[str, Any]] = {}
                with open(inference_rl_log_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        sample_key = str(
                            row.get("sample_index_raw", row.get("sample_index"))
                        )
                        restored_rows_by_sample[sample_key] = row
                restored_rows = list(restored_rows_by_sample.values())
                inference_rl_running_sample_count = len(restored_rows)
                inference_rl_running_correct_count = sum(
                    1 for row in restored_rows if bool(row.get("final_correct", False))
                )
            except Exception as e:
                logger.warning(
                    f"[InferenceRL] failed to restore running counters from log: {e}"
                )
        os.makedirs(os.path.dirname(inference_rl_log_path), exist_ok=True)
        logger.info(
            f"[InferenceRL] enabled. task={task_name_str}, algorithm={args.rl_algorithm}, "
            f"action_set={args.rl_action_set}, verify_mode={args.rl_verify_mode}, "
            f"max_decisions={args.rl_max_decisions_per_sample}, "
            f"max_attempts={args.rl_max_attempts}, "
            f"max_retrieval_injections={args.rl_max_retrieval_injections}, "
            f"max_verifications={args.rl_max_verifications}, "
            f"budget_target={args.rl_budget_target}, cost_lambda={args.rl_cost_lambda}, "
            f"gamma={args.rl_gamma}, lr={args.rl_lr}, "
            f"epsilon_start={args.rl_epsilon_start}, epsilon_end={args.rl_epsilon_end}, "
            f"target_update_interval={args.rl_target_update_interval}, "
            f"replay_buffer_size={args.rl_replay_buffer_size}, "
            f"actions={list(OS_RL_ACTIONS)}"
        )
    if bandit_enabled:
        from src.controllers.dbbench_linucb import (
            GeneralFeatureAdapter,
            LinUCB,
            TwoHeadLinUCB,
            DBBenchFeatureAdapter,
            GENERAL_BUDGET_PRIMITIVES,
            TOOL_FRIENDLY_BUDGET_PRIMITIVES,
            REPLAY_GATED_BUDGET_PRIMITIVES,
            RETRIEVAL_GATED_BUDGET_PRIMITIVES,
            RETRIEVAL_SIZE_GATED_BUDGET_PRIMITIVES,
            REPLAY_HEAVY_RETRIEVAL_SIZE_GATED_BUDGET_PRIMITIVES,
        )

        if (
            task_name_str == "db_bench"
            and args.bandit_replay_heavy_retrieval_size_gated
        ):
            dbbench_budget_primitives = list(
                REPLAY_HEAVY_RETRIEVAL_SIZE_GATED_BUDGET_PRIMITIVES
            )
        elif task_name_str == "db_bench" and args.bandit_retrieval_size_gated:
            dbbench_budget_primitives = list(RETRIEVAL_SIZE_GATED_BUDGET_PRIMITIVES)
        elif args.bandit_retrieval_gated:
            dbbench_budget_primitives = list(RETRIEVAL_GATED_BUDGET_PRIMITIVES)
        elif args.bandit_replay_gated:
            dbbench_budget_primitives = list(REPLAY_GATED_BUDGET_PRIMITIVES)
        elif task_name_str == "db_bench" and args.bandit_decoupled_action_space:
            dbbench_budget_primitives = list(TOOL_FRIENDLY_BUDGET_PRIMITIVES)
        else:
            dbbench_budget_primitives = list(GENERAL_BUDGET_PRIMITIVES)
        selected_bandit_feature_names = [
            feature_name.strip()
            for feature_name in str(args.bandit_feature_subset).split(",")
            if feature_name.strip()
        ]
        if args.bandit_budget_only_baseline:
            retrieval_only_feature_names = {
                "retrieved_example_count",
                "top1_similarity_score",
                "avg_similarity_score",
                "avg_rounds_in_retrieved",
                "avg_cost_in_retrieved",
            }
            if selected_bandit_feature_names:
                selected_bandit_feature_names = [
                    feature_name
                    for feature_name in selected_bandit_feature_names
                    if feature_name not in retrieval_only_feature_names
                ]
            else:
                selected_bandit_feature_names = list(GENERALIZABLE_FEATURE_NAMES)
        if task_name_str == "db_bench":
            dbbench_feature_adapter = DBBenchFeatureAdapter(
                use_richer_features=bool(args.bandit_richer_features),
                selected_feature_names=selected_bandit_feature_names or None,
            )
        else:
            if args.bandit_richer_features:
                logger.warning(
                    "[Bandit] richer DBBench-only features requested on task=%s; ignoring.",
                    task_name_str,
                )
            dbbench_feature_adapter = GeneralFeatureAdapter(
                selected_feature_names=selected_bandit_feature_names or None,
            )
        bandit_action_counts = [0 for _ in dbbench_budget_primitives]
        bandit_action_cost_ema_normalized = [None for _ in dbbench_budget_primitives]
        bandit_action_cost_ema_usd = [None for _ in dbbench_budget_primitives]
        rescue_action_counts = [0 for _ in dbbench_budget_primitives]
        rescue_profile_idx = next(
            (
                idx
                for idx, profile in enumerate(dbbench_budget_primitives)
                if profile.name == args.bandit_rescue_profile_name
            ),
            max(
                range(len(dbbench_budget_primitives)),
                key=lambda idx: dbbench_budget_primitives[idx].static_cost_proxy,
            ),
        )
        max_budget_profile_idx = max(
            range(len(dbbench_budget_primitives)),
            key=lambda idx: dbbench_budget_primitives[idx].static_cost_proxy,
        )
        min_budget_profile_idx = min(
            range(len(dbbench_budget_primitives)),
            key=lambda idx: dbbench_budget_primitives[idx].static_cost_proxy,
        )
        allowed_policy_action_indices = list(range(len(dbbench_budget_primitives)))
        if not args.bandit_allow_max_arm:
            allowed_policy_action_indices = [
                idx
                for idx in allowed_policy_action_indices
                if idx != max_budget_profile_idx
            ]
            # Safety fallback to avoid empty action set if config changes.
            if len(allowed_policy_action_indices) == 0:
                allowed_policy_action_indices = [max_budget_profile_idx]
        if args.bandit_disallow_lowest_arm:
            allowed_policy_action_indices = [
                idx
                for idx in allowed_policy_action_indices
                if idx != min_budget_profile_idx
            ]
            if len(allowed_policy_action_indices) == 0:
                allowed_policy_action_indices = [min_budget_profile_idx]
        if args.bandit_two_head:
            dbbench_bandit = TwoHeadLinUCB(
                n_actions=len(dbbench_budget_primitives),
                feature_dim=dbbench_feature_adapter.feature_dim,
                alpha=float(args.bandit_alpha),
            )
        else:
            dbbench_bandit = LinUCB(
                n_actions=len(dbbench_budget_primitives),
                feature_dim=dbbench_feature_adapter.feature_dim,
                alpha=float(args.bandit_alpha),
            )
        os.makedirs(os.path.dirname(bandit_log_path), exist_ok=True)
        bandit_rescue_enabled = bool(
            args.bandit_rescue_enable
            and task_name_str == "db_bench"
            and (
                args.bandit_retrieval_size_gated
                or args.bandit_replay_heavy_retrieval_size_gated
            )
        )
        logger.info(
            f"[Bandit] enabled. task={task_name_str}, alpha={args.bandit_alpha}, lambda_init={args.bandit_lambda}, "
            f"two_head={args.bandit_two_head}, adaptive_lambda={args.bandit_adaptive_lambda}, "
            f"budget_only_baseline={args.bandit_budget_only_baseline}, "
            f"richer_features={args.bandit_richer_features}, "
            f"feature_names={list(dbbench_feature_adapter.feature_names)}, "
            f"residual_updates={args.bandit_residual_updates}, "
            f"decoupled_action_space={args.bandit_decoupled_action_space}, "
            f"two_timescale={args.bandit_two_timescale}, correct_head_lr={correct_head_lr_scale}, "
            f"cost_head_lr={cost_head_lr_scale}, "
            f"warm_start_slow_adaptation={args.bandit_warm_start_slow_adaptation}, "
            f"warm_start_samples={warm_start_samples}, post_warmup_scale={post_warmup_scale}, "
            f"budget_target={args.bandit_budget_target}, lambda_lr={args.bandit_lambda_lr}, "
            f"policy={args.bandit_policy}, threshold_init={args.bandit_correctness_threshold}, "
            f"threshold_adaptive={args.bandit_adaptive_threshold}, target_acc={args.bandit_target_accuracy}, "
            f"acc_tol={args.bandit_accuracy_tolerance}, threshold_step={args.bandit_threshold_step}, "
            f"cost_ema_decay={args.bandit_cost_ema_decay}, "
            f"rescue_enable={bandit_rescue_enabled}, "
            f"rescue_profile={dbbench_budget_primitives[rescue_profile_idx].name}, "
            f"rescue_on_incorrect={args.bandit_rescue_on_incorrect}, "
            f"rescue_on_task_limit={args.bandit_rescue_on_task_limit}, "
            f"rescue_on_incomplete={args.bandit_rescue_on_incomplete}, "
            f"rescue_confidence_gate={args.bandit_rescue_confidence_gate}, "
            f"rescue_predicted_correct_max={args.bandit_rescue_predicted_correct_max}, "
            f"rescue_uncertainty_min={args.bandit_rescue_uncertainty_min}, "
            f"threshold_min={args.bandit_threshold_min}, threshold_max={args.bandit_threshold_max}, "
            f"allow_max_arm={args.bandit_allow_max_arm}, "
            f"disallow_lowest_arm={args.bandit_disallow_lowest_arm}, "
            f"min_predicted_correct_floor={args.bandit_min_predicted_correct_floor}, "
            f"cost_guardrail_enable={args.bandit_cost_guardrail_enable}, "
            f"cost_guardrail_band={args.bandit_cost_guardrail_band}, "
            f"actions={[p.name for p in dbbench_budget_primitives]}"
        )
    if art_enabled:
        from src.controllers.dbbench_art_lora import (
            DBBenchARTConfig,
            DBBenchARTPolicy,
            resolve_art_action_set,
        )

        if not args.dbbench_art_adapter_path:
            raise ValueError(
                "DBBench ART+LoRA runtime requires --dbbench_art_adapter_path."
            )
        dbbench_budget_primitives = list(resolve_art_action_set(args.dbbench_art_action_set))
        allocator_base_model = str(
            args.dbbench_art_allocator_base_model
            or args.dbbench_art_base_model
            or getattr(getattr(agent, "_language_model", None), "model_name_or_path", "")
            or getattr(llm_obj, "model_name_or_path", "")
            or getattr(llm_obj, "model_id", "")
            or getattr(llm_obj, "name", "")
            or getattr(llm_obj, "model", "")
            or "unknown"
        )
        dbbench_art_config = DBBenchARTConfig(
            base_model=allocator_base_model,
            allowed_primitives=list(dbbench_budget_primitives),
            reward_lambda=float(args.dbbench_art_reward_lambda),
            budget_target_usd=max(float(args.dbbench_art_budget_target), 1e-9),
            output_dir=assignment_config.output_dir,
            gpu_memory_utilization=float(args.dbbench_art_gpu_memory_utilization),
            max_model_len=int(args.dbbench_art_max_model_len),
        )
        dbbench_art_policy = DBBenchARTPolicy(
            config=dbbench_art_config,
            adapter_path=str(args.dbbench_art_adapter_path),
            cost_tracker=cost_tracker,
        )
        dbbench_art_manifest = dict(getattr(dbbench_art_policy, "manifest_payload", {}))
        dbbench_art_action_counts = {
            primitive.name: 0 for primitive in dbbench_budget_primitives
        }
        os.makedirs(os.path.dirname(dbbench_art_log_path), exist_ok=True)
        logger.info(
            f"[DBBenchART] enabled. adapter_path={args.dbbench_art_adapter_path}, "
            f"allocator_base_model={allocator_base_model}, reward_lambda={args.dbbench_art_reward_lambda}, "
            f"budget_target={args.dbbench_art_budget_target}, action_set={args.dbbench_art_action_set}, "
            f"gpu_memory_utilization={args.dbbench_art_gpu_memory_utilization}, "
            f"max_model_len={args.dbbench_art_max_model_len}, "
            f"policy_mode={dbbench_art_manifest.get('policy_mode', 'unknown')}, "
            f"actions={[p.name for p in dbbench_budget_primitives]}"
        )

    logger.info(
        f"Experiment start. "
        f"Total sample count: {len(assignment_config.sample_order)}. "
        f"Unfinished sample count: {len(unfinished_sample_order)}."
    )

    def apply_bandit_profile(profile: Any) -> None:
        set_generation_token_budget(agent, profile.token_budget)
        task.max_round = profile.max_round
        if hasattr(task, "tool_budget"):
            setattr(task, "tool_budget", int(profile.tool_budget))
        if hasattr(task, "stop_enabled"):
            setattr(task, "stop_enabled", bool(profile.stop_enabled))

    def set_task_replay_controls(
        replay_mode: str,
        replay_sample_count: int | None,
    ) -> None:
        task._copal_replay_mode = replay_mode  # type: ignore[attr-defined]
        task._copal_replay_sample_count = replay_sample_count  # type: ignore[attr-defined]

    def get_retrieval_feature_summary(
        *,
        current_sample_index: Any,
        replay_sample_count: int | None = None,
    ) -> dict[str, Any]:
        for callback in callback_dict.values():
            summary_getter = getattr(callback, "get_retrieval_feature_summary", None)
            if callable(summary_getter):
                summary = summary_getter(
                    task=task,
                    current_sample_index=current_sample_index,
                    replay_sample_count=replay_sample_count,
                )
                if isinstance(summary, dict):
                    summary = dict(summary)
                    avg_cost_usd = float(summary.get("avg_cost_in_retrieved_usd", 0.0))
                    summary["avg_cost_in_retrieved_normalized"] = (
                        avg_cost_usd / bandit_budget_target
                        if bandit_budget_target > 0.0
                        else 0.0
                    )
                    return summary
        return {
            "retrieved_example_count": 0,
            "top1_similarity_score": 0.0,
            "avg_similarity_score": 0.0,
            "avg_rounds_in_retrieved": 0.0,
            "avg_cost_in_retrieved_usd": 0.0,
            "avg_cost_in_retrieved_normalized": 0.0,
            "selected_sample_indices": [],
        }

    def select_bandit_profile(
        dataset_item: Any,
        *,
        sample_index: Any,
        chat_history: Any | None,
    ) -> tuple[Any, list[float], int, dict[str, Any], list[int], bool, list[float] | None]:
        assert dbbench_bandit is not None
        assert dbbench_feature_adapter is not None
        retrieval_feature_summary = get_retrieval_feature_summary(
            current_sample_index=sample_index,
        )
        features = dbbench_feature_adapter.build(
            dataset_item,
            chat_history=chat_history,
            system_prompt=str(getattr(agent, "_system_prompt", "")),
            language_model=getattr(agent, "_language_model", None),
            runtime_state={
                "running_cost_ema_normalized": (
                    running_cost_per_sample_ema_normalized or 0.0
                ),
                "running_accuracy": (
                    float(bandit_running_correct_count)
                    / float(bandit_running_sample_count)
                    if bandit_running_sample_count > 0
                    else 0.0
                ),
                "previous_sample_cost_normalized": previous_sample_cost_normalized,
                "previous_sample_correct": previous_sample_correct,
                "previous_sample_had_error": previous_sample_had_error,
                "previous_action_uncertainty": previous_action_uncertainty,
                "recent_task_limit_rate": recent_task_limit_rate,
                "cost_regime_delta": (
                    0.0
                    if running_cost_short_ema_normalized is None
                    or running_cost_long_ema_normalized is None
                    else (
                        running_cost_short_ema_normalized
                        - running_cost_long_ema_normalized
                    )
                ),
                "replay_advantage_ema": replay_advantage_ema,
                "successful_task_token_sets": successful_task_token_sets,
                "retrieval_feature_summary": retrieval_feature_summary,
            },
        )
        candidate_indices = list(allowed_policy_action_indices)
        guardrail_over_budget = False
        if (
            args.bandit_cost_guardrail_enable
            and running_cost_per_sample_ema_normalized is not None
            and running_cost_per_sample_ema_normalized
            > (1.0 + float(args.bandit_cost_guardrail_band))
        ):
            guardrail_over_budget = True
            if len(candidate_indices) > 1:
                max_candidate_idx = max(
                    candidate_indices,
                    key=lambda idx: dbbench_budget_primitives[idx].static_cost_proxy,
                )
                candidate_indices = [
                    idx for idx in candidate_indices if idx != max_candidate_idx
                ]
        cost_hints: list[float] | None = None
        if args.bandit_two_head:
            if args.bandit_policy in {"cheapest_safe", "fixed_tau"}:
                min_proxy = min(
                    p.static_cost_proxy for p in dbbench_budget_primitives
                )
                cost_hints = []
                for idx, primitive in enumerate(dbbench_budget_primitives):
                    ema_cost = bandit_action_cost_ema_normalized[idx]
                    if ema_cost is not None:
                        cost_hints.append(ema_cost)
                    else:
                        proxy_scale = primitive.static_cost_proxy / min_proxy
                        cost_hints.append(proxy_scale)
                if args.bandit_policy == "cheapest_safe":
                    action_idx, action_meta = dbbench_bandit.select_cheapest_safe_action(
                        features,
                        correctness_threshold=current_correctness_threshold,
                        action_cost_hint=cost_hints,
                        lambda_value=current_bandit_lambda,
                        candidate_action_indices=candidate_indices,
                    )
                else:
                    action_idx, action_meta = dbbench_bandit.select_fixed_tau_action(
                        features,
                        correctness_threshold=current_correctness_threshold,
                        action_cost_hint=cost_hints,
                        candidate_action_indices=candidate_indices,
                    )
            else:
                action_idx, action_meta = dbbench_bandit.select_action(
                    features,
                    lambda_value=current_bandit_lambda,
                    min_predicted_correct=max(
                        0.0, float(args.bandit_min_predicted_correct_floor)
                    ),
                    candidate_action_indices=candidate_indices,
                )
        else:
            action_idx, ucb_score = dbbench_bandit.select_action(features)
            action_meta = {"score": ucb_score}
        profile = dbbench_budget_primitives[action_idx]
        action_meta = dict(action_meta)
        action_meta["retrieval_feature_summary"] = retrieval_feature_summary
        return (
            profile,
            features,
            action_idx,
            action_meta,
            candidate_indices,
            guardrail_over_budget,
            cost_hints,
        )

    def _normalize_signature_text(text: Any) -> str:
        return " ".join(str(text or "").strip().split())

    def _build_trace_signature(session: Session) -> str:
        trace_segments: list[str] = []
        for item_index in range(session.chat_history.get_value_length()):
            item = session.chat_history.get_item_deep_copy(item_index)
            if item.role != Role.AGENT:
                continue
            raw_content = str(item.content or "")
            try:
                parsed = task._parse_agent_response(raw_content)  # type: ignore[attr-defined]
                action_name = getattr(parsed.action, "value", str(parsed.action))
                if parsed.content is not None:
                    content_str = _normalize_signature_text(parsed.content)
                else:
                    content_str = _normalize_signature_text(parsed.finish_reason)
                trace_segments.append(f"{action_name}:{content_str}")
            except Exception:
                trace_segments.append(f"raw:{_normalize_signature_text(raw_content)}")
        if trace_segments:
            return " || ".join(trace_segments)
        return f"empty:{session.sample_status.value}"

    def _get_cost_tracker_calls() -> list[Any]:
        return list(getattr(cost_tracker, "calls", None) or [])

    def _get_call_window_cost_usd(start_idx: int, end_idx: int) -> float:
        calls = _get_cost_tracker_calls()
        return float(sum(call.total_cost_usd for call in calls[start_idx:end_idx]))

    def _get_self_consistency_inference_override() -> dict[str, Any]:
        return {
            "do_sample": bool(args.self_consistency_do_sample),
            "temperature": float(args.self_consistency_temperature),
            "top_p": float(args.self_consistency_top_p),
        }

    def _set_agent_inference_config_override(
        override: Optional[Mapping[str, Any]],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        had_attr = hasattr(agent, "_inference_config_dict")
        previous_config = copy.deepcopy(
            getattr(agent, "_inference_config_dict", None)
        )
        if override is not None:
            merged_config = (
                copy.deepcopy(previous_config)
                if isinstance(previous_config, dict)
                else {}
            )
            merged_config.update({key: value for key, value in override.items()})
            setattr(agent, "_inference_config_dict", merged_config)
        return had_attr, previous_config

    def _restore_agent_inference_config_override(
        had_attr: bool,
        previous_config: Optional[dict[str, Any]],
    ) -> None:
        if had_attr:
            setattr(agent, "_inference_config_dict", previous_config)
        elif hasattr(agent, "_inference_config_dict"):
            delattr(agent, "_inference_config_dict")

    def _select_heuristic_replay_controls(
        dataset_item: Any,
        *,
        sample_index: Any,
    ) -> tuple[str, int | None, dict[str, Any]]:
        replay_sample_count = int(args.heuristic_replay_sample_count)
        retrieval_feature_summary = get_retrieval_feature_summary(
            current_sample_index=sample_index,
            replay_sample_count=replay_sample_count,
        )
        skill_count = 0
        skill_list_getter = getattr(dataset_item, "get_skill_list", None)
        if callable(skill_list_getter):
            try:
                skill_count = len(skill_list_getter())
            except Exception:
                skill_count = len(getattr(dataset_item, "skill_list", []) or [])
        else:
            skill_count = len(getattr(dataset_item, "skill_list", []) or [])
        similarity_score = float(
            retrieval_feature_summary.get("top1_similarity_score", 0.0) or 0.0
        )
        trigger_map = {
            "similarity": similarity_score
            >= float(args.heuristic_replay_similarity_threshold),
            "skill_count": skill_count >= int(args.heuristic_replay_skill_threshold),
            "recent_task_limit_rate": recent_task_limit_rate
            >= float(args.heuristic_replay_task_limit_threshold),
        }
        if args.heuristic_replay_rule == "all":
            should_replay = all(trigger_map.values())
        else:
            should_replay = any(trigger_map.values())
        replay_mode = str(args.heuristic_replay_mode) if should_replay else "none"
        replay_count = replay_sample_count if should_replay else None
        metadata = {
            "rule": str(args.heuristic_replay_rule),
            "trigger_map": trigger_map,
            "skill_count": skill_count,
            "recent_task_limit_rate": float(recent_task_limit_rate),
            "replay_mode": replay_mode,
            "replay_sample_count": replay_count,
            "retrieval_feature_summary": retrieval_feature_summary,
        }
        return replay_mode, replay_count, metadata

    def _run_non_bandit_session(
        *,
        sample_index: Any,
        replay_mode: str,
        replay_sample_count: int | None,
        inference_override: Optional[Mapping[str, Any]] = None,
        fire_task_complete_callback: bool = True,
    ) -> tuple[Session, CallbackArguments]:
        had_override_attr, previous_override = _set_agent_inference_config_override(
            inference_override
        )
        try:
            set_task_replay_controls(replay_mode, replay_sample_count)
            current_session = Session(task_name=task.task_name, sample_index=sample_index)
            current_callback_args = CallbackArguments(
                current_session=current_session,
                task=task,
                agent=agent,
                session_list=session_list,
            )
            callback_handler.on_session_create(current_callback_args)
            if current_callback_args.session_controller.should_task_reset:
                task.reset(current_session)
                callback_handler.on_task_reset(current_callback_args)
            while current_session.sample_status == SampleStatus.RUNNING:
                if current_callback_args.session_controller.should_agent_inference:
                    agent.inference(current_session)
                    callback_handler.on_agent_inference(current_callback_args)
                if current_callback_args.session_controller.should_task_interact:
                    task.interact(current_session)
                    callback_handler.on_task_interact(current_callback_args)
            if current_callback_args.session_controller.should_task_complete:
                task.complete(current_session)
                if fire_task_complete_callback:
                    callback_handler.on_task_complete(current_callback_args)
            return current_session, current_callback_args
        finally:
            _restore_agent_inference_config_override(
                had_override_attr, previous_override
            )

    for sample_index in unfinished_sample_order:
        if inference_rl_enabled:
            assert inference_rl_controller is not None
            logger.info(f"Sample {sample_index} start.")
            inference_rl_runtime_state = {
                "running_accuracy": (
                    float(inference_rl_running_correct_count)
                    / float(inference_rl_running_sample_count)
                    if inference_rl_running_sample_count > 0
                    else 0.0
                ),
                "recent_task_limit_rate": recent_task_limit_rate,
                "previous_sample_cost_normalized": previous_sample_cost_normalized,
                "previous_sample_correct": previous_sample_correct,
            }
            rl_result = run_os_inference_rl_episode(
                sample_index=sample_index,
                task=task,
                agent=agent,
                callback_handler=callback_handler,
                callback_dict=callback_dict,
                session_list=session_list,
                cost_tracker=cost_tracker,
                controller=inference_rl_controller,
                output_dir=assignment_config.output_dir,
                runtime_state=inference_rl_runtime_state,
                max_decisions_per_sample=int(args.rl_max_decisions_per_sample),
                max_attempts=int(args.rl_max_attempts),
                max_retrieval_injections=int(args.rl_max_retrieval_injections),
                max_verifications=int(args.rl_max_verifications),
                budget_target_usd=max(float(args.rl_budget_target), 1e-9),
                cost_lambda=float(args.rl_cost_lambda),
                verifier_mode=str(args.rl_verify_mode),
            )
            session = rl_result.session
            callback_args = rl_result.callback_args
            session_list.append(session)
            json.dump(
                [s.model_dump() for s in session_list],
                open(session_list_output_path, "w"),  # noqa
                indent=2,
            )
            logger.info(
                f"Sample {sample_index} end. Session status: {session.sample_status}. "
                f"Evaluation outcome: {session.evaluation_record.outcome}. "
                f"[InferenceRL] reward={rl_result.summary['reward']:.6f}, "
                f"cost={rl_result.summary['total_cost_usd']:.6f}, "
                f"decisions={rl_result.summary['decision_count']}, "
                f"retries={rl_result.summary['retry_count']}."
            )
            inference_rl_running_sample_count += 1
            if session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT:
                inference_rl_running_correct_count += 1
            sample_cost_normalized = float(
                rl_result.summary.get("total_cost_normalized", 0.0) or 0.0
            )
            previous_sample_cost_normalized = sample_cost_normalized
            previous_sample_correct = (
                session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
            )
            previous_sample_had_error = bool(
                session.sample_status
                in (
                    SampleStatus.AGENT_VALIDATION_FAILED,
                    SampleStatus.TASK_LIMIT_REACHED,
                    SampleStatus.TASK_ENVIRONMENT_ERROR,
                    SampleStatus.TASK_UNKNOWN_ERROR,
                    SampleStatus.AGENT_CONTEXT_LIMIT,
                    SampleStatus.AGENT_OUT_OF_MEMORY,
                    SampleStatus.AGENT_UNKNOWN_ERROR,
                )
            )
            task_limit_indicator = (
                1.0 if session.sample_status == SampleStatus.TASK_LIMIT_REACHED else 0.0
            )
            recent_task_limit_rate = (0.8 * recent_task_limit_rate) + (
                0.2 * task_limit_indicator
            )
            callback_handler.on_state_save(callback_args)
            continue

        if args.self_consistency_enable or args.heuristic_replay_gate_enable:
            logger.info(f"Sample {sample_index} start.")
            dataset_item_for_baseline = task.get_dataset_item_for_sample(sample_index)
            if args.self_consistency_enable:
                os.makedirs(os.path.dirname(self_consistency_log_path), exist_ok=True)
                total_attempt_cost = 0.0
                attempt_records: list[dict[str, Any]] = []
                selected_session: Session | None = None
                selected_callback_args: CallbackArguments | None = None
                for attempt_index in range(int(args.self_consistency_max_attempts)):
                    if args.heuristic_replay_gate_enable:
                        replay_mode, replay_sample_count, heuristic_metadata = (
                            _select_heuristic_replay_controls(
                                dataset_item_for_baseline,
                                sample_index=sample_index,
                            )
                        )
                    else:
                        replay_mode, replay_sample_count, heuristic_metadata = (
                            str(args.self_consistency_replay_mode),
                            (
                                int(args.self_consistency_replay_sample_count)
                                if str(args.self_consistency_replay_mode)
                                == "retrieved"
                                else None
                            ),
                            {
                                "self_consistency_replay_mode": str(
                                    args.self_consistency_replay_mode
                                ),
                                "self_consistency_replay_sample_count": int(
                                    args.self_consistency_replay_sample_count
                                ),
                            },
                        )
                    start_call_idx = len(_get_cost_tracker_calls())
                    attempt_session, attempt_callback_args = _run_non_bandit_session(
                        sample_index=sample_index,
                        replay_mode=replay_mode,
                        replay_sample_count=replay_sample_count,
                        inference_override=_get_self_consistency_inference_override(),
                        fire_task_complete_callback=False,
                    )
                    end_call_idx = len(_get_cost_tracker_calls())
                    attempt_cost_usd = _get_call_window_cost_usd(
                        start_call_idx, end_call_idx
                    )
                    total_attempt_cost += attempt_cost_usd
                    attempt_signature = _build_trace_signature(attempt_session)
                    attempt_record = {
                        "attempt_index": attempt_index,
                        "session": attempt_session,
                        "callback_args": attempt_callback_args,
                        "cost_usd": attempt_cost_usd,
                        "signature": attempt_signature,
                        "heuristic_metadata": heuristic_metadata,
                        "sample_status": attempt_session.sample_status.value,
                        "evaluation_outcome": attempt_session.evaluation_record.outcome.value,
                    }
                    attempt_records.append(attempt_record)
                    logger.info(
                        f"[SelfConsistency] sample={sample_index}, attempt={attempt_index}, "
                        f"signature={attempt_signature}, cost_usd={attempt_cost_usd:.6f}, "
                        f"status={attempt_session.sample_status.value}, "
                        f"outcome={attempt_session.evaluation_record.outcome.value}, "
                        f"replay_mode={replay_mode}, replay_sample_count={replay_sample_count}"
                    )
                    if (
                        float(args.self_consistency_cost_budget_usd) > 0.0
                        and total_attempt_cost
                        >= float(args.self_consistency_cost_budget_usd)
                    ):
                        break
                signature_counter = Counter(
                    record["signature"] for record in attempt_records
                )
                selected_record = min(
                    attempt_records,
                    key=lambda record: (
                        -signature_counter[record["signature"]],
                        float(record["cost_usd"]),
                        int(record["attempt_index"]),
                    ),
                )
                selected_session = selected_record["session"]
                selected_callback_args = selected_record["callback_args"]
                callback_handler.on_task_complete(selected_callback_args)
                session = selected_session
                callback_args = selected_callback_args
                with open(self_consistency_log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sample_index": sample_index,
                                "selected_attempt_index": int(
                                    selected_record["attempt_index"]
                                ),
                                "selected_signature": str(
                                    selected_record["signature"]
                                ),
                                "attempt_count": len(attempt_records),
                                "total_attempt_cost_usd": float(total_attempt_cost),
                                "cost_budget_usd": float(
                                    args.self_consistency_cost_budget_usd
                                ),
                                "attempts": [
                                    {
                                        "attempt_index": int(record["attempt_index"]),
                                        "cost_usd": float(record["cost_usd"]),
                                        "signature": str(record["signature"]),
                                        "sample_status": str(record["sample_status"]),
                                        "evaluation_outcome": str(
                                            record["evaluation_outcome"]
                                        ),
                                        "heuristic_metadata": record[
                                            "heuristic_metadata"
                                        ],
                                    }
                                    for record in attempt_records
                                ],
                            }
                        )
                        + "\n"
                    )
                logger.info(
                    f"[SelfConsistency] sample={sample_index}, selected_attempt={selected_record['attempt_index']}, "
                    f"consensus_size={signature_counter[selected_record['signature']]}, "
                    f"attempt_count={len(attempt_records)}, total_attempt_cost_usd={total_attempt_cost:.6f}"
                )
            else:
                replay_mode, replay_sample_count, heuristic_metadata = (
                    _select_heuristic_replay_controls(
                        dataset_item_for_baseline,
                        sample_index=sample_index,
                    )
                )
                os.makedirs(os.path.dirname(heuristic_log_path), exist_ok=True)
                with open(heuristic_log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sample_index": sample_index,
                                "replay_mode": replay_mode,
                                "replay_sample_count": replay_sample_count,
                                "heuristic_metadata": heuristic_metadata,
                            }
                        )
                        + "\n"
                    )
                logger.info(
                    f"[HeuristicGate] sample={sample_index}, replay_mode={replay_mode}, "
                    f"replay_sample_count={replay_sample_count}, metadata={heuristic_metadata}"
                )
                session, callback_args = _run_non_bandit_session(
                    sample_index=sample_index,
                    replay_mode=replay_mode,
                    replay_sample_count=replay_sample_count,
                    inference_override=None,
                    fire_task_complete_callback=True,
                )

            session_list.append(session)
            json.dump(
                [s.model_dump() for s in session_list],
                open(session_list_output_path, "w"),  # noqa
                indent=2,
            )
            logger.info(
                f"Sample {sample_index} end. Session status: {session.sample_status}. "
                f"Evaluation outcome: {session.evaluation_record.outcome}."
            )
            task_limit_indicator = (
                1.0 if session.sample_status == SampleStatus.TASK_LIMIT_REACHED else 0.0
            )
            recent_task_limit_rate = (0.8 * recent_task_limit_rate) + (
                0.2 * task_limit_indicator
            )
            callback_handler.on_state_save(callback_args)
            continue

        preselected_bandit_state: dict[str, Any] | None = None
        if (
            bandit_enabled
            and dbbench_bandit is not None
            and dbbench_feature_adapter is not None
            and bool(
                args.bandit_replay_gated
                or args.bandit_retrieval_gated
                or args.bandit_retrieval_size_gated
                or args.bandit_replay_heavy_retrieval_size_gated
            )
        ):
            dataset_item_for_routing = task.get_dataset_item_for_sample(sample_index)
            (
                pre_profile,
                pre_features,
                pre_action_idx,
                pre_action_meta,
                pre_candidate_indices,
                pre_guardrail_over_budget,
                pre_cost_hints,
            ) = select_bandit_profile(
                dataset_item_for_routing,
                sample_index=sample_index,
                chat_history=None,
            )
            set_task_replay_controls(
                getattr(pre_profile, "replay_mode", "full"),
                getattr(pre_profile, "replay_sample_count", None),
            )
            preselected_bandit_state = {
                "profile": pre_profile,
                "features": pre_features,
                "action_idx": pre_action_idx,
                "action_meta": pre_action_meta,
                "candidate_indices": pre_candidate_indices,
                "guardrail_over_budget": pre_guardrail_over_budget,
                "cost_hints": pre_cost_hints,
            }
        else:
            set_task_replay_controls("full", None)

        # region Initialize session
        session = Session(task_name=task.task_name, sample_index=sample_index)
        callback_args = CallbackArguments(
            current_session=session, task=task, agent=agent, session_list=session_list
        )
        callback_handler.on_session_create(callback_args)
        if callback_args.session_controller.should_task_reset:
            task.reset(session)
            callback_handler.on_task_reset(callback_args)
        if (
            bandit_enabled
            and dbbench_bandit is not None
            and dbbench_feature_adapter is not None
        ):
            sample_key = str(sample_index)
            if preselected_bandit_state is not None:
                profile = preselected_bandit_state["profile"]
                features = preselected_bandit_state["features"]
                action_idx = int(preselected_bandit_state["action_idx"])
                action_meta = preselected_bandit_state["action_meta"]
                candidate_indices = preselected_bandit_state["candidate_indices"]
                guardrail_over_budget = preselected_bandit_state[
                    "guardrail_over_budget"
                ]
                cost_hints = preselected_bandit_state["cost_hints"]
            else:
                dataset_item = task._get_current_dataset_item()  # noqa: SLF001
                (
                    profile,
                    features,
                    action_idx,
                    action_meta,
                    candidate_indices,
                    guardrail_over_budget,
                    cost_hints,
                ) = select_bandit_profile(
                    dataset_item,
                    sample_index=sample_index,
                    chat_history=session.chat_history,
                )
            bandit_action_counts[action_idx] += 1

            # Apply per-sample compute budget.
            apply_bandit_profile(profile)

            # CoPAL-L replay gate: write the chosen replay controls onto the
            # task so the replay callback can read them in on_session_create.
            set_task_replay_controls(
                getattr(profile, "replay_mode", "full"),
                getattr(profile, "replay_sample_count", None),
            )

            # Save info to update bandit after completion.
            calls = getattr(cost_tracker, "calls", None) or []
            bandit_state_by_sample[sample_key] = {
                "features": features,
                "feature_names": list(dbbench_feature_adapter.feature_names),
                "action_idx": action_idx,
                "profile_name": profile.name,
                "token_budget": profile.token_budget,
                "max_round": profile.max_round,
                "tool_budget": profile.tool_budget,
                "stop_enabled": profile.stop_enabled,
                "replay_mode": getattr(profile, "replay_mode", "full"),
                "replay_sample_count": getattr(profile, "replay_sample_count", None),
                "start_call_idx": len(calls),
                "action_meta": action_meta,
                "lambda_before": current_bandit_lambda,
                "threshold_before": current_correctness_threshold,
                "cost_hint": (
                    cost_hints[action_idx]
                    if args.bandit_two_head
                    and args.bandit_policy in {"cheapest_safe", "fixed_tau"}
                    else None
                ),
                "rescue_triggered": False,
                "rescue_reason": None,
                "rescue_profile_name": None,
                "rescue_replay_mode": None,
                "rescue_replay_sample_count": None,
                "candidate_indices": candidate_indices,
                "guardrail_over_budget": guardrail_over_budget,
            }
            logger.info(
                f"[Bandit] sample={sample_index}, action={profile.name}, "
                f"replay_mode={getattr(profile, 'replay_mode', 'full')}, "
                f"replay_sample_count={getattr(profile, 'replay_sample_count', None)}, "
                f"token_budget={profile.token_budget}, max_round={profile.max_round}, "
                f"tool_budget={profile.tool_budget}, stop_enabled={profile.stop_enabled}, "
                f"lambda={current_bandit_lambda:.6f}, threshold={current_correctness_threshold:.4f}, "
                f"guardrail_over_budget={guardrail_over_budget}, "
                f"action_meta={action_meta}"
            )
        elif (
            art_enabled
            and dbbench_art_policy is not None
            and dbbench_art_config is not None
        ):
            sample_key = str(sample_index)
            from src.controllers.dbbench_art_lora import build_dbbench_art_scenario

            dataset_item = task._get_current_dataset_item()  # noqa: SLF001
            scenario = build_dbbench_art_scenario(
                sample_index=sample_index,
                dataset_item=dataset_item,
                allowed_primitives=dbbench_budget_primitives,
            )
            calls = getattr(cost_tracker, "calls", None) or []
            start_call_idx = len(calls)
            profile = dbbench_art_policy.select_primitive(scenario)
            dbbench_art_action_counts[profile.name] = (
                dbbench_art_action_counts.get(profile.name, 0) + 1
            )
            set_generation_token_budget(agent, profile.token_budget)
            task.max_round = profile.max_round
            if hasattr(task, "tool_budget"):
                setattr(task, "tool_budget", int(profile.tool_budget))
            if hasattr(task, "stop_enabled"):
                setattr(task, "stop_enabled", bool(profile.stop_enabled))
            art_state_by_sample[sample_key] = {
                "scenario": scenario,
                "profile_name": profile.name,
                "token_budget": profile.token_budget,
                "max_round": profile.max_round,
                "tool_budget": profile.tool_budget,
                "stop_enabled": profile.stop_enabled,
                "selection_meta": dict(getattr(dbbench_art_policy, "last_selection_meta", {})),
                "start_call_idx": start_call_idx,
            }
            logger.info(
                f"[DBBenchART] sample={sample_index}, action={profile.name}, "
                f"token_budget={profile.token_budget}, max_round={profile.max_round}, "
                f"tool_budget={profile.tool_budget}, stop_enabled={profile.stop_enabled}, "
                f"selection_meta={art_state_by_sample[sample_key]['selection_meta']}"
            )
        logger.info(f"Sample {sample_index} start.")
        # endregion
        # region Run session
        while session.sample_status == SampleStatus.RUNNING:
            if callback_args.session_controller.should_agent_inference:
                agent.inference(session)
                callback_handler.on_agent_inference(callback_args)
            if callback_args.session_controller.should_task_interact:
                task.interact(session)
                callback_handler.on_task_interact(callback_args)
        # endregion
        # region Complete session
        if callback_args.session_controller.should_task_complete:
            task.complete(session)
            callback_handler.on_task_complete(callback_args)
        original_session = session
        original_callback_args = callback_args
        if bandit_enabled and dbbench_bandit is not None:
            bandit_state = bandit_state_by_sample.get(str(sample_index))
            if bandit_state is not None:
                calls = getattr(cost_tracker, "calls", None) or []
                bandit_state["original_end_call_idx"] = len(calls)
                bandit_state["original_status"] = original_session.sample_status.value
                bandit_state["original_outcome"] = (
                    original_session.evaluation_record.outcome.value
                )
        if bandit_enabled and dbbench_bandit is not None:
            should_rescue = False
            rescue_reason: Optional[str] = None
            bandit_state = bandit_state_by_sample.get(str(sample_index))
            original_profile_name = (
                str(bandit_state.get("profile_name"))
                if bandit_state is not None
                else None
            )
            original_action_meta = (
                bandit_state.get("action_meta", {}) if bandit_state is not None else {}
            )
            original_predicted_correct = original_action_meta.get("predicted_correct")
            original_uncertainty = original_action_meta.get("uncertainty")
            if original_session.sample_status != SampleStatus.COMPLETED:
                status_key = original_session.sample_status.value
                non_completed_status_count[status_key] = (
                    non_completed_status_count.get(status_key, 0) + 1
                )
            already_max_retrieval_arm = (
                original_profile_name == str(args.bandit_rescue_profile_name)
            )
            if not already_max_retrieval_arm:
                if (
                    args.bandit_rescue_on_task_limit
                    and original_session.sample_status == SampleStatus.TASK_LIMIT_REACHED
                ):
                    should_rescue = True
                    rescue_reason = "status=task_limit_reached"
                elif (
                    args.bandit_rescue_confidence_gate
                    and isinstance(original_predicted_correct, (int, float))
                    and float(original_predicted_correct)
                    < float(args.bandit_rescue_predicted_correct_max)
                ):
                    should_rescue = True
                    rescue_reason = (
                        "predicted_correct="
                        f"{float(original_predicted_correct):.4f}"
                    )
                elif (
                    args.bandit_rescue_confidence_gate
                    and isinstance(original_uncertainty, (int, float))
                    and float(original_uncertainty)
                    > float(args.bandit_rescue_uncertainty_min)
                ):
                    should_rescue = True
                    rescue_reason = (
                        "uncertainty="
                        f"{float(original_uncertainty):.4f}"
                    )
                elif (
                    args.bandit_rescue_on_incomplete
                    and original_session.sample_status != SampleStatus.COMPLETED
                ):
                    should_rescue = True
                    rescue_reason = f"status={original_session.sample_status.value}"
                elif (
                    args.bandit_rescue_on_incorrect
                    and original_session.sample_status == SampleStatus.COMPLETED
                    and original_session.evaluation_record.outcome.value != "correct"
                ):
                    should_rescue = True
                    rescue_reason = (
                        "completed_but_incorrect:"
                        f"{original_session.evaluation_record.outcome.value}"
                    )

            if should_rescue and bandit_rescue_enabled:
                rescue_profile = dbbench_budget_primitives[rescue_profile_idx]
                rescue_action_counts[rescue_profile_idx] += 1
                bandit_rescue_count += 1
                bandit_state = bandit_state_by_sample.get(str(sample_index))
                if bandit_state is not None:
                    bandit_state["rescue_triggered"] = True
                    bandit_state["rescue_reason"] = rescue_reason
                    bandit_state["rescue_profile_name"] = rescue_profile.name
                    bandit_state["rescue_replay_mode"] = getattr(
                        rescue_profile, "replay_mode", "full"
                    )
                    bandit_state["rescue_replay_sample_count"] = getattr(
                        rescue_profile, "replay_sample_count", None
                    )
                    calls = getattr(cost_tracker, "calls", None) or []
                    bandit_state["rescue_start_call_idx"] = len(calls)
                logger.info(
                    f"[Bandit] rescue start sample={sample_index}, reason={rescue_reason}, "
                    f"profile={rescue_profile.name}, token_budget={rescue_profile.token_budget}, "
                    f"max_round={rescue_profile.max_round}, "
                    f"replay_mode={getattr(rescue_profile, 'replay_mode', 'full')}, "
                    f"replay_sample_count={getattr(rescue_profile, 'replay_sample_count', None)}"
                )

                rescue_session = Session(
                    task_name=task.task_name, sample_index=sample_index
                )
                rescue_callback_args = CallbackArguments(
                    current_session=rescue_session,
                    task=task,
                    agent=agent,
                    session_list=session_list,
                )
                set_task_replay_controls(
                    getattr(rescue_profile, "replay_mode", "full"),
                    getattr(rescue_profile, "replay_sample_count", None),
                )
                callback_handler.on_session_create(rescue_callback_args)
                if rescue_callback_args.session_controller.should_task_reset:
                    task.reset(rescue_session)
                    callback_handler.on_task_reset(rescue_callback_args)

                apply_bandit_profile(rescue_profile)

                while rescue_session.sample_status == SampleStatus.RUNNING:
                    if rescue_callback_args.session_controller.should_agent_inference:
                        agent.inference(rescue_session)
                        callback_handler.on_agent_inference(rescue_callback_args)
                    if rescue_callback_args.session_controller.should_task_interact:
                        task.interact(rescue_session)
                        callback_handler.on_task_interact(rescue_callback_args)

                if rescue_callback_args.session_controller.should_task_complete:
                    task.complete(rescue_session)
                    callback_handler.on_task_complete(rescue_callback_args)

                if rescue_session.evaluation_record.outcome.value == "correct":
                    bandit_rescue_success_count += 1
                if bandit_state is not None:
                    calls = getattr(cost_tracker, "calls", None) or []
                    bandit_state["rescue_end_call_idx"] = len(calls)
                    bandit_state["final_status"] = rescue_session.sample_status.value
                    bandit_state["final_outcome"] = (
                        rescue_session.evaluation_record.outcome.value
                    )
                logger.info(
                    f"[Bandit] rescue end sample={sample_index}, "
                    f"status={rescue_session.sample_status}, outcome={rescue_session.evaluation_record.outcome}"
                )
                session = rescue_session
                callback_args = rescue_callback_args
            else:
                bandit_state = bandit_state_by_sample.get(str(sample_index))
                if bandit_state is not None:
                    bandit_state["rescue_start_call_idx"] = bandit_state.get(
                        "original_end_call_idx", bandit_state["start_call_idx"]
                    )
                    bandit_state["rescue_end_call_idx"] = bandit_state[
                        "rescue_start_call_idx"
                    ]
                    bandit_state["final_status"] = original_session.sample_status.value
                    bandit_state["final_outcome"] = (
                        original_session.evaluation_record.outcome.value
                    )
        session_list.append(session)
        json.dump(
            [s.model_dump() for s in session_list],
            open(session_list_output_path, "w"),  # noqa
            indent=2,
        )
        logger.info(
            f"Sample {sample_index} end. Session status: {session.sample_status}. "
            f"Evaluation outcome: {session.evaluation_record.outcome}."
        )
        if bandit_enabled and dbbench_bandit is not None:
            bandit_state = bandit_state_by_sample.get(str(sample_index))
            if bandit_state is not None:
                calls = getattr(cost_tracker, "calls", None) or []
                start_idx = int(bandit_state["start_call_idx"])
                original_end_idx = int(
                    bandit_state.get("original_end_call_idx", len(calls))
                )
                rescue_end_idx = int(
                    bandit_state.get("rescue_end_call_idx", len(calls))
                )
                original_calls = calls[start_idx:original_end_idx]
                rescue_calls = calls[original_end_idx:rescue_end_idx]
                original_cost = float(sum(c.total_cost_usd for c in original_calls))
                rescue_cost = float(sum(c.total_cost_usd for c in rescue_calls))
                sample_cost = original_cost + rescue_cost
                original_cost_normalized = original_cost / bandit_budget_target
                rescue_cost_normalized = rescue_cost / bandit_budget_target
                sample_cost_normalized = sample_cost / bandit_budget_target
                original_correct = bandit_state.get("original_outcome") == "correct"
                final_correct = session.evaluation_record.outcome.value == "correct"
                reward_lambda = float(bandit_state["lambda_before"])
                reward = (1.0 if original_correct else 0.0) - (
                    reward_lambda * original_cost_normalized
                )
                adaptation_scale = 1.0
                if (
                    args.bandit_warm_start_slow_adaptation
                    and bandit_running_sample_count >= warm_start_samples
                ):
                    adaptation_scale = post_warmup_scale
                if args.bandit_two_head:
                    dbbench_bandit.update(
                        action_idx=int(bandit_state["action_idx"]),
                        x=bandit_state["features"],
                        correct_label=1.0 if original_correct else 0.0,
                        cost_label=original_cost_normalized,
                        correct_lr_scale=(correct_head_lr_scale * adaptation_scale),
                        cost_lr_scale=(cost_head_lr_scale * adaptation_scale),
                        use_residual=bool(args.bandit_residual_updates),
                    )
                else:
                    dbbench_bandit.update(
                        action_idx=int(bandit_state["action_idx"]),
                        x=bandit_state["features"],
                        reward=reward,
                        lr_scale=adaptation_scale,
                    )
                action_idx = int(bandit_state["action_idx"])
                prev_norm_ema = bandit_action_cost_ema_normalized[action_idx]
                prev_usd_ema = bandit_action_cost_ema_usd[action_idx]
                if prev_norm_ema is None:
                    bandit_action_cost_ema_normalized[action_idx] = (
                        original_cost_normalized
                    )
                else:
                    decay = float(args.bandit_cost_ema_decay)
                    bandit_action_cost_ema_normalized[action_idx] = (
                        decay * prev_norm_ema
                    ) + ((1.0 - decay) * original_cost_normalized)
                if prev_usd_ema is None:
                    bandit_action_cost_ema_usd[action_idx] = original_cost
                else:
                    decay = float(args.bandit_cost_ema_decay)
                    bandit_action_cost_ema_usd[action_idx] = (decay * prev_usd_ema) + (
                        (1.0 - decay) * original_cost
                    )
                if args.bandit_adaptive_lambda:
                    current_bandit_lambda = min(
                        1.0,
                        max(
                            0.0,
                            current_bandit_lambda
                            + (
                                (float(args.bandit_lambda_lr) * adaptation_scale)
                                * (sample_cost_normalized - 1.0)
                            ),
                        ),
                    )
                else:
                    current_bandit_lambda = min(1.0, max(0.0, current_bandit_lambda))
                bandit_lambda_history.append(current_bandit_lambda)
                if running_cost_per_sample_ema_usd is None:
                    running_cost_per_sample_ema_usd = sample_cost
                else:
                    guardrail_decay = float(args.bandit_cost_guardrail_decay)
                    running_cost_per_sample_ema_usd = (
                        guardrail_decay * running_cost_per_sample_ema_usd
                    ) + ((1.0 - guardrail_decay) * sample_cost)
                if running_cost_per_sample_ema_normalized is None:
                    running_cost_per_sample_ema_normalized = sample_cost_normalized
                else:
                    guardrail_decay = float(args.bandit_cost_guardrail_decay)
                    running_cost_per_sample_ema_normalized = (
                        guardrail_decay * running_cost_per_sample_ema_normalized
                    ) + ((1.0 - guardrail_decay) * sample_cost_normalized)
                short_decay = 0.5
                long_decay = 0.9
                if running_cost_short_ema_normalized is None:
                    running_cost_short_ema_normalized = sample_cost_normalized
                else:
                    running_cost_short_ema_normalized = (
                        short_decay * running_cost_short_ema_normalized
                    ) + ((1.0 - short_decay) * sample_cost_normalized)
                if running_cost_long_ema_normalized is None:
                    running_cost_long_ema_normalized = sample_cost_normalized
                else:
                    running_cost_long_ema_normalized = (
                        long_decay * running_cost_long_ema_normalized
                    ) + ((1.0 - long_decay) * sample_cost_normalized)
                bandit_running_sample_count += 1
                if final_correct:
                    bandit_running_correct_count += 1
                previous_sample_cost_normalized = sample_cost_normalized
                previous_sample_correct = bool(final_correct)
                previous_sample_had_error = bool(
                    session.sample_status
                    in (
                        SampleStatus.AGENT_VALIDATION_FAILED,
                        SampleStatus.TASK_LIMIT_REACHED,
                        SampleStatus.TASK_ENVIRONMENT_ERROR,
                        SampleStatus.TASK_UNKNOWN_ERROR,
                        SampleStatus.AGENT_CONTEXT_LIMIT,
                        SampleStatus.AGENT_OUT_OF_MEMORY,
                        SampleStatus.AGENT_UNKNOWN_ERROR,
                    )
                )
                previous_action_uncertainty = float(
                    bandit_state.get("action_meta", {}).get("uncertainty", 0.0) or 0.0
                )
                task_limit_indicator = (
                    1.0
                    if session.sample_status == SampleStatus.TASK_LIMIT_REACHED
                    else 0.0
                )
                recent_task_limit_rate = (0.8 * recent_task_limit_rate) + (
                    0.2 * task_limit_indicator
                )
                replay_mode_for_sample = str(bandit_state.get("replay_mode", "full"))
                action_used_replay = replay_mode_for_sample != "none"
                if action_used_replay:
                    if replay_reward_ema is None:
                        replay_reward_ema = reward
                    else:
                        replay_reward_ema = (0.8 * replay_reward_ema) + (0.2 * reward)
                else:
                    if no_replay_reward_ema is None:
                        no_replay_reward_ema = reward
                    else:
                        no_replay_reward_ema = (0.8 * no_replay_reward_ema) + (
                            0.2 * reward
                        )
                if replay_reward_ema is not None and no_replay_reward_ema is not None:
                    replay_advantage_ema = min(
                        max(replay_reward_ema - no_replay_reward_ema, -1.0),
                        1.0,
                    )
                if final_correct:
                    dataset_item_for_memory = task.get_dataset_item_for_sample(sample_index)
                    item_text = dbbench_feature_adapter._extract_item_text(  # type: ignore[attr-defined]
                        dataset_item_for_memory
                    )
                    token_set = dbbench_feature_adapter._tokenize_text(item_text)  # type: ignore[attr-defined]
                    if token_set:
                        successful_task_token_sets.append(token_set)
                        if len(successful_task_token_sets) > 128:
                            successful_task_token_sets.pop(0)
                running_accuracy = (
                    float(bandit_running_correct_count)
                    / float(bandit_running_sample_count)
                    if bandit_running_sample_count > 0
                    else 0.0
                )
                if args.bandit_adaptive_threshold:
                    acc_target = float(args.bandit_target_accuracy)
                    acc_tol = float(args.bandit_accuracy_tolerance)
                    threshold_step = (
                        float(args.bandit_threshold_step) * adaptation_scale
                    )
                    threshold_min = float(args.bandit_threshold_min)
                    threshold_max = float(args.bandit_threshold_max)
                    if running_accuracy < (acc_target - acc_tol):
                        current_correctness_threshold = min(
                            threshold_max,
                            current_correctness_threshold + threshold_step,
                        )
                    elif running_accuracy > (acc_target + acc_tol):
                        current_correctness_threshold = max(
                            threshold_min,
                            current_correctness_threshold - threshold_step,
                        )
                    if current_correctness_threshold <= threshold_min:
                        threshold_min_hit_count += 1
                    if current_correctness_threshold >= threshold_max:
                        threshold_max_hit_count += 1
                bandit_threshold_history.append(current_correctness_threshold)
                os.makedirs(os.path.dirname(bandit_log_path), exist_ok=True)
                with open(bandit_log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sample_index": sample_index,
                                "sample_index_raw": str(sample_index),
                                "task_name": task_name_str,
                                "correct": final_correct,
                                "original_correct": bool(original_correct),
                                "sample_cost_usd": sample_cost,
                                "original_cost_usd": original_cost,
                                "rescue_cost_usd": rescue_cost,
                                "sample_cost_normalized": sample_cost_normalized,
                                "original_cost_normalized": original_cost_normalized,
                                "rescue_cost_normalized": rescue_cost_normalized,
                                "reward": reward,
                                "action_idx": int(bandit_state["action_idx"]),
                                "profile_name": bandit_state["profile_name"],
                                "token_budget": int(bandit_state["token_budget"]),
                                "max_round": int(bandit_state["max_round"]),
                                "tool_budget": int(bandit_state["tool_budget"]),
                                "stop_enabled": bool(bandit_state["stop_enabled"]),
                                "replay_mode": bandit_state.get("replay_mode"),
                                "replay_sample_count": bandit_state.get(
                                    "replay_sample_count"
                                ),
                                "action_meta": bandit_state.get("action_meta", {}),
                                "feature_names": bandit_state.get("feature_names", []),
                                "feature_values": {
                                    feature_name: float(feature_value)
                                    for feature_name, feature_value in zip(
                                        bandit_state.get("feature_names", []),
                                        bandit_state["features"],
                                    )
                                },
                                "lambda_before": float(bandit_state["lambda_before"]),
                                "lambda_after": float(current_bandit_lambda),
                                "threshold_before": float(
                                    bandit_state["threshold_before"]
                                ),
                                "threshold_after": float(current_correctness_threshold),
                                "running_accuracy": float(running_accuracy),
                                "cost_hint": bandit_state.get("cost_hint"),
                                "budget_target": float(args.bandit_budget_target),
                                "candidate_indices": bandit_state.get(
                                    "candidate_indices"
                                ),
                                "guardrail_over_budget": bool(
                                    bandit_state.get("guardrail_over_budget", False)
                                ),
                                "running_cost_per_sample_ema_usd": (
                                    None
                                    if running_cost_per_sample_ema_usd is None
                                    else float(running_cost_per_sample_ema_usd)
                                ),
                                "running_cost_per_sample_normalized_ema": (
                                    None
                                    if running_cost_per_sample_ema_normalized is None
                                    else float(running_cost_per_sample_ema_normalized)
                                ),
                                "rescue_triggered": bool(
                                    bandit_state.get("rescue_triggered", False)
                                ),
                                "rescue_reason": bandit_state.get("rescue_reason"),
                                "rescue_profile_name": bandit_state.get(
                                    "rescue_profile_name"
                                ),
                                "rescue_replay_mode": bandit_state.get(
                                    "rescue_replay_mode"
                                ),
                                "rescue_replay_sample_count": bandit_state.get(
                                    "rescue_replay_sample_count"
                                ),
                            }
                        )
                        + "\n"
                    )
                logger.info(
                    f"[Bandit] sample={sample_index}, final_correct={final_correct}, "
                    f"original_correct={original_correct}, original_cost={original_cost:.6f}, "
                    f"rescue_cost={rescue_cost:.6f}, total_cost={sample_cost:.6f}, "
                    f"total_cost_normalized={sample_cost_normalized:.4f}, reward={reward:.6f}, "
                    f"lambda_after={current_bandit_lambda:.6f}, "
                    f"threshold_after={current_correctness_threshold:.4f}, "
                    f"running_accuracy={running_accuracy:.4f}, "
                    f"running_cost_ema_usd={0.0 if running_cost_per_sample_ema_usd is None else running_cost_per_sample_ema_usd:.6f}, "
                    f"running_cost_ema_normalized={0.0 if running_cost_per_sample_ema_normalized is None else running_cost_per_sample_ema_normalized:.4f}"
                )
        if art_enabled and dbbench_art_config is not None:
            art_state = art_state_by_sample.get(str(sample_index))
            if art_state is not None:
                calls = getattr(cost_tracker, "calls", None) or []
                start_idx = int(art_state["start_call_idx"])
                sample_calls = calls[start_idx:]
                sample_cost = float(sum(c.total_cost_usd for c in sample_calls))
                prompt_tokens = int(sum(c.prompt_tokens for c in sample_calls))
                completion_tokens = int(sum(c.completion_tokens for c in sample_calls))
                sample_cost_normalized = sample_cost / max(
                    float(dbbench_art_config.budget_target_usd), 1e-9
                )
                correct = session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
                reward = (1.0 if correct else 0.0) - (
                    float(dbbench_art_config.reward_lambda) * sample_cost_normalized
                )
                selection_meta = dict(art_state.get("selection_meta", {}))
                with open(dbbench_art_log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "sample_index": sample_index,
                                "action_name": art_state["profile_name"],
                                "correct": bool(correct),
                                "sample_status": session.sample_status.value,
                                "cost_usd": sample_cost,
                                "cost_normalized": sample_cost_normalized,
                                "reward": reward,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "adapter_path": str(args.dbbench_art_adapter_path),
                                "policy_version": str(
                                    dbbench_art_manifest.get(
                                        "policy_version", "dbbench_art_lora_v1"
                                    )
                                ),
                                "selection_meta": selection_meta,
                            }
                        )
                        + "\n"
                    )
                logger.info(
                    f"[DBBenchART] sample={sample_index}, correct={bool(correct)}, "
                    f"sample_cost={sample_cost:.6f}, sample_cost_normalized={sample_cost_normalized:.4f}, "
                    f"reward={reward:.6f}"
                )
        # endregion
        # region Save callback state
        # The state of callback will be used to restore the previous incomplete assignment.
        callback_handler.on_state_save(callback_args)
        # endregion
    # endregion
    # region Evaluate
    session_metric_calculation_partial_list: Sequence[
        SessionMetricCalculationPartial
    ] = [
        SessionMetricCalculationPartial(
            sample_index=session.sample_index,
            evaluation_record=session.evaluation_record,
            sample_status=session.sample_status,
        )
        for session in session_list
    ]
    metric = task.calculate_metric(session_metric_calculation_partial_list)
    cost_summary = cost_tracker.summary()
    sample_count = len(assignment_config.sample_order)
    mean_cost_per_sample = (
        float(cost_summary["total_cost_usd"]) / sample_count
        if sample_count > 0
        else 0.0
    )
    success_card_extraction_summary = None
    for callback in callback_dict.values():
        summary = getattr(callback, "success_card_extraction_summary", None)
        if summary is not None:
            success_card_extraction_summary = dict(summary)
            break

    extraction_prompt_tokens = 0
    extraction_completion_tokens = 0
    extraction_total_tokens = 0
    extraction_cost_usd = 0.0
    if success_card_extraction_summary is not None:
        extraction_prompt_tokens = int(
            success_card_extraction_summary.get("total_prompt_tokens", 0)
        )
        extraction_completion_tokens = int(
            success_card_extraction_summary.get("total_completion_tokens", 0)
        )
        extraction_total_tokens = (
            extraction_prompt_tokens + extraction_completion_tokens
        )
        extraction_cost_usd = float(
            success_card_extraction_summary.get("total_cost_usd", 0.0)
        )

    inference_only_prompt_tokens = max(
        0, int(cost_summary["total_prompt_tokens"]) - extraction_prompt_tokens
    )
    inference_only_completion_tokens = max(
        0,
        int(cost_summary["total_completion_tokens"]) - extraction_completion_tokens,
    )
    inference_only_total_tokens = (
        inference_only_prompt_tokens + inference_only_completion_tokens
    )
    inference_only_cost_usd = max(
        0.0, float(cost_summary["total_cost_usd"]) - extraction_cost_usd
    )
    inference_only_mean_cost_per_sample = (
        inference_only_cost_usd / sample_count if sample_count > 0 else 0.0
    )

    metric["cost"] = {
        "total_prompt_tokens": int(cost_summary["total_prompt_tokens"]),
        "total_completion_tokens": int(cost_summary["total_completion_tokens"]),
        "total_tokens": int(cost_summary["total_tokens"]),
        "total_cost_usd": float(cost_summary["total_cost_usd"]),
        "mean_cost_per_sample_usd": mean_cost_per_sample,
        "inference_only_prompt_tokens": inference_only_prompt_tokens,
        "inference_only_completion_tokens": inference_only_completion_tokens,
        "inference_only_total_tokens": inference_only_total_tokens,
        "inference_only_cost_usd": inference_only_cost_usd,
        "inference_only_mean_cost_per_sample_usd": inference_only_mean_cost_per_sample,
        "success_card_extraction_prompt_tokens": extraction_prompt_tokens,
        "success_card_extraction_completion_tokens": extraction_completion_tokens,
        "success_card_extraction_total_tokens": extraction_total_tokens,
        "success_card_extraction_cost_usd": extraction_cost_usd,
    }
    if success_card_extraction_summary is not None:
        metric["success_card_extraction"] = success_card_extraction_summary
    logger.info(
        f"Experiment end. Metric: {metric}. Total sample count: {len(assignment_config.sample_order)}.",
    )
    logger.info(
        "[RunCostSummary] "
        f"input_tokens={metric['cost']['total_prompt_tokens']}, "
        f"output_tokens={metric['cost']['total_completion_tokens']}, "
        f"total_tokens={metric['cost']['total_tokens']}, "
        f"total_cost_usd={metric['cost']['total_cost_usd']:.6f}, "
        f"mean_cost_per_sample_usd={metric['cost']['mean_cost_per_sample_usd']:.6f}"
    )
    if success_card_extraction_summary is not None:
        logger.info(
            "[RunCostSummary:InferenceOnly] "
            f"input_tokens={metric['cost']['inference_only_prompt_tokens']}, "
            f"output_tokens={metric['cost']['inference_only_completion_tokens']}, "
            f"total_tokens={metric['cost']['inference_only_total_tokens']}, "
            f"total_cost_usd={metric['cost']['inference_only_cost_usd']:.6f}, "
            f"mean_cost_per_sample_usd={metric['cost']['inference_only_mean_cost_per_sample_usd']:.6f}"
        )
        logger.info(
            "[RunCostSummary:SuccessCardExtraction] "
            f"input_tokens={metric['cost']['success_card_extraction_prompt_tokens']}, "
            f"output_tokens={metric['cost']['success_card_extraction_completion_tokens']}, "
            f"total_tokens={metric['cost']['success_card_extraction_total_tokens']}, "
            f"total_cost_usd={metric['cost']['success_card_extraction_cost_usd']:.6f}"
        )
    if bandit_enabled:
        try:
            bandit_log_entries = []
            if os.path.exists(bandit_log_path):
                with open(bandit_log_path, "r") as f:
                    bandit_log_entries = [
                        json.loads(line) for line in f if line.strip()
                    ]
            total = len(bandit_log_entries)
            mean_cost = (
                sum(row["sample_cost_usd"] for row in bandit_log_entries) / total
                if total > 0
                else 0.0
            )
            mean_cost_normalized = (
                sum(
                    row.get(
                        "sample_cost_normalized",
                        float(row["sample_cost_usd"]) / bandit_budget_target,
                    )
                    for row in bandit_log_entries
                )
                / total
                if total > 0
                else 0.0
            )
            accuracy = (
                sum(1 for row in bandit_log_entries if row["correct"]) / total
                if total > 0
                else 0.0
            )
            original_accuracy = (
                sum(
                    1
                    for row in bandit_log_entries
                    if row.get("original_correct", False)
                )
                / total
                if total > 0
                else 0.0
            )
            pass_rate = accuracy
            cost_of_pass = (mean_cost / pass_rate) if pass_rate > 0 else None
            metric["dbbench_bandit"] = {
                "enabled": True,
                "task_name": task_name_str,
                "alpha": float(args.bandit_alpha),
                "lambda_init": float(args.bandit_lambda),
                "lambda_final": float(current_bandit_lambda),
                "lambda_adaptive": bool(args.bandit_adaptive_lambda),
                "lambda_lr": float(args.bandit_lambda_lr),
                "budget_target": float(args.bandit_budget_target),
                "two_head": bool(args.bandit_two_head),
                "richer_features": bool(args.bandit_richer_features),
                "feature_subset": str(args.bandit_feature_subset),
                "feature_names": (
                    list(dbbench_feature_adapter.feature_names)
                    if dbbench_feature_adapter is not None
                    else []
                ),
                "residual_updates": bool(args.bandit_residual_updates),
                "decoupled_action_space": bool(args.bandit_decoupled_action_space),
                "two_timescale": bool(args.bandit_two_timescale),
                "correct_head_lr": float(correct_head_lr_scale),
                "cost_head_lr": float(cost_head_lr_scale),
                "warm_start_slow_adaptation": bool(
                    args.bandit_warm_start_slow_adaptation
                ),
                "warm_start_samples": int(warm_start_samples),
                "post_warmup_scale": float(post_warmup_scale),
                "policy": str(args.bandit_policy),
                "correctness_threshold_init": float(args.bandit_correctness_threshold),
                "correctness_threshold_final": float(current_correctness_threshold),
                "threshold_adaptive": bool(args.bandit_adaptive_threshold),
                "threshold_min": float(args.bandit_threshold_min),
                "threshold_max": float(args.bandit_threshold_max),
                "samples": total,
                "action_counts": {
                    dbbench_budget_primitives[idx].name: int(count)
                    for idx, count in enumerate(bandit_action_counts)
                },
                "rescue_action_counts": {
                    dbbench_budget_primitives[idx].name: int(count)
                    for idx, count in enumerate(rescue_action_counts)
                },
                "rescue_count": int(bandit_rescue_count),
                "rescue_success_count": int(bandit_rescue_success_count),
                "rescue_enabled": bool(bandit_rescue_enabled),
                "rescue_confidence_gate": bool(args.bandit_rescue_confidence_gate),
                "rescue_predicted_correct_max": float(
                    args.bandit_rescue_predicted_correct_max
                ),
                "rescue_uncertainty_min": float(args.bandit_rescue_uncertainty_min),
                "action_cost_ema_usd": {
                    dbbench_budget_primitives[idx].name: (
                        None if ema is None else float(ema)
                    )
                    for idx, ema in enumerate(bandit_action_cost_ema_usd)
                },
                "action_cost_ema_normalized": {
                    dbbench_budget_primitives[idx].name: (
                        None if ema is None else float(ema)
                    )
                    for idx, ema in enumerate(bandit_action_cost_ema_normalized)
                },
                "mean_cost_usd": mean_cost,
                "mean_cost_normalized": mean_cost_normalized,
                "accuracy": accuracy,
                "original_accuracy": original_accuracy,
                "cost_of_pass": cost_of_pass,
                "running_cost_per_sample_ema_usd_final": (
                    None
                    if running_cost_per_sample_ema_usd is None
                    else float(running_cost_per_sample_ema_usd)
                ),
                "running_cost_per_sample_ema_normalized_final": (
                    None
                    if running_cost_per_sample_ema_normalized is None
                    else float(running_cost_per_sample_ema_normalized)
                ),
                "cost_guardrail_enable": bool(args.bandit_cost_guardrail_enable),
                "cost_guardrail_band": float(args.bandit_cost_guardrail_band),
                "allow_max_arm": bool(args.bandit_allow_max_arm),
                "disallow_lowest_arm": bool(args.bandit_disallow_lowest_arm),
                "min_predicted_correct_floor": float(
                    args.bandit_min_predicted_correct_floor
                ),
                "normal_action_set": [
                    dbbench_budget_primitives[idx].name
                    for idx in allowed_policy_action_indices
                ],
                "mean_lambda": (
                    sum(bandit_lambda_history) / len(bandit_lambda_history)
                    if bandit_lambda_history
                    else float(args.bandit_lambda)
                ),
                "mean_threshold": (
                    sum(bandit_threshold_history) / len(bandit_threshold_history)
                    if bandit_threshold_history
                    else float(args.bandit_correctness_threshold)
                ),
                "threshold_min_hit_rate": (
                    float(threshold_min_hit_count) / float(total) if total > 0 else 0.0
                ),
                "threshold_max_hit_rate": (
                    float(threshold_max_hit_count) / float(total) if total > 0 else 0.0
                ),
                "non_completed_status_count": non_completed_status_count,
            }
            metric["contextual_bandit"] = dict(metric["dbbench_bandit"])
        except Exception as e:
            logger.error(f"[Bandit] failed to append summary metric: {e}")
    if art_enabled and dbbench_art_config is not None:
        try:
            art_log_entries = []
            if os.path.exists(dbbench_art_log_path):
                with open(dbbench_art_log_path, "r") as f:
                    art_log_entries = [json.loads(line) for line in f if line.strip()]
            total = len(art_log_entries)
            accuracy = (
                sum(1 for row in art_log_entries if row["correct"]) / total
                if total > 0
                else 0.0
            )
            mean_cost = (
                sum(float(row["cost_usd"]) for row in art_log_entries) / total
                if total > 0
                else 0.0
            )
            mean_cost_normalized = (
                sum(float(row["cost_normalized"]) for row in art_log_entries) / total
                if total > 0
                else 0.0
            )
            metric["dbbench_art_lora"] = {
                "enabled": True,
                "policy_version": str(
                    dbbench_art_manifest.get("policy_version", "dbbench_art_lora_v1")
                ),
                "allocator_base_model": str(
                    dbbench_art_manifest.get(
                        "allocator_base_model",
                        dbbench_art_manifest.get("base_model", dbbench_art_config.base_model),
                    )
                ),
                "base_model": str(
                    dbbench_art_manifest.get(
                        "allocator_base_model",
                        dbbench_art_manifest.get("base_model", dbbench_art_config.base_model),
                    )
                ),
                "adapter_path": str(args.dbbench_art_adapter_path),
                "project": dbbench_art_manifest.get("project"),
                "art_model_name": dbbench_art_manifest.get("art_model_name"),
                "action_counts": dict(dbbench_art_action_counts),
                "accuracy": accuracy,
                "mean_cost_usd": mean_cost,
                "mean_cost_normalized": mean_cost_normalized,
                "cost_of_pass": (mean_cost / accuracy) if accuracy > 0 else None,
                "reward_lambda": float(
                    dbbench_art_manifest.get(
                        "reward_lambda", dbbench_art_config.reward_lambda
                    )
                ),
                "budget_target_usd": float(
                    dbbench_art_manifest.get(
                        "budget_target_usd", dbbench_art_config.budget_target_usd
                    )
                ),
                "group_size": int(
                    dbbench_art_manifest.get("group_size", dbbench_art_config.group_size)
                ),
                "temperature": float(
                    dbbench_art_manifest.get(
                        "temperature", dbbench_art_config.temperature
                    )
                ),
                "lora_rank": int(
                    dbbench_art_manifest.get("lora_rank", dbbench_art_config.lora_rank)
                ),
                "epochs": int(
                    dbbench_art_manifest.get(
                        "epochs", dbbench_art_config.max_training_epochs
                    )
                ),
                "training_scenarios": int(
                    dbbench_art_manifest.get("training_scenarios", 0)
                ),
                "evaluation_scenarios": int(total),
                "action_set": dbbench_art_manifest.get(
                    "action_set", str(args.dbbench_art_action_set)
                ),
                "backend_path": dbbench_art_manifest.get("backend_path"),
                "latest_step": dbbench_art_manifest.get("latest_step"),
            }
        except Exception as e:
            logger.error(f"[DBBenchART] failed to append summary metric: {e}")
    if inference_rl_enabled:
        try:
            from src.controllers.inference_allocation_rl import (
                write_inference_rl_summary,
            )

            inference_rl_summary = write_inference_rl_summary(
                log_path=inference_rl_log_path,
                summary_path=inference_rl_summary_path,
                controller=inference_rl_controller,
            )
            inference_rl_summary.update(
                {
                    "budget_target": float(args.rl_budget_target),
                    "cost_lambda": float(args.rl_cost_lambda),
                    "gamma": float(args.rl_gamma),
                    "lr": float(args.rl_lr),
                    "epsilon_start": float(args.rl_epsilon_start),
                    "epsilon_end": float(args.rl_epsilon_end),
                    "target_update_interval": int(args.rl_target_update_interval),
                    "replay_buffer_size": int(args.rl_replay_buffer_size),
                    "max_decisions_per_sample": int(
                        args.rl_max_decisions_per_sample
                    ),
                    "max_attempts": int(args.rl_max_attempts),
                    "max_retrieval_injections": int(
                        args.rl_max_retrieval_injections
                    ),
                    "max_verifications": int(args.rl_max_verifications),
                    "verify_mode": str(args.rl_verify_mode),
                    "action_set": str(args.rl_action_set),
                }
            )
            metric["inference_rl"] = inference_rl_summary
        except Exception as e:
            logger.error(f"[InferenceRL] failed to append summary metric: {e}")
    json.dump(
        metric,
        open(path_config.metric_output_path, "w"),  # noqa
        indent=2,
    )
    logger.info(f"Metric file has been saved to {assignment_config.output_dir}.")
    # endregion
    # --------------------------------------
    # CostTracker save
    # --------------------------------------
    try:
        from pathlib import Path

        run_dir = Path(
            assignment_config.output_dir
        )  # same directory metrics are already saved to
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        cost_tracker.save(str(run_dir / "metrics" / "token_cost_summary.json"))
        logger.info(
            f"[CostTracker] wrote {run_dir/'metrics'/'token_cost_summary.json'}"
        )
    except Exception as e:
        logger.error(f"[CostTracker] failed to save summary: {e}")
    # --------------------------------------
    # region Release
    task.release()
    # endregion


if __name__ == "__main__":
    main()
