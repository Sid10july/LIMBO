from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Optional, Sequence

from src.controllers.dbbench_linucb import BudgetPrimitive, TOOL_FRIENDLY_BUDGET_PRIMITIVES
from src.metrics.cost_tracker import CostTracker
from src.tasks.instance.db_bench.task import DBBenchDatasetItem
from src.utils import set_generation_token_budget
from src.typings import Session, SessionEvaluationOutcome, SampleIndex, SampleStatus


ART_POLICY_VERSION = "dbbench_art_lora_v2"
ART_FALLBACK_ACTION = "budget_384_r2_t4"
ART_PROJECT = "copal-dbbench"
ART_SYSTEM_PROMPT = (
    "You are a compute allocator. Choose exactly one budget profile for a DBBench "
    'SQL task. Output only JSON with key "action".'
)


@dataclass(frozen=True)
class DBBenchARTScenario:
    sample_index: SampleIndex
    sample_index_int: Optional[int]
    instruction: str
    table_name: str
    column_names: list[str]
    row_count: int
    column_count: int
    skill_list: list[str]
    allowed_actions: list[str]
    ground_truth_sql_type: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DBBenchARTRolloutResult:
    sample_index: SampleIndex
    action_name: str
    correct: bool
    sample_status: str
    cost_usd: float
    cost_normalized: float
    reward: float
    prompt_tokens: int
    completion_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DBBenchARTConfig:
    base_model: str
    allowed_primitives: list[BudgetPrimitive]
    reward_lambda: float = 0.3
    budget_target_usd: float = 0.00009
    group_size: int = 6
    temperature: float = 0.7
    lora_rank: int = 8
    lora_alpha: int = 16
    max_training_epochs: int = 3
    max_rollouts_per_scenario: int = 6
    output_dir: str = "outputs"
    backend: str = "local"
    strict_json: bool = True
    use_manifest_fallback: bool = False
    local_backend_path: Optional[str] = None
    max_allocator_tokens: int = 64
    groups_per_step: int = 1
    art_learning_rate: float = 5e-6
    baseline_mean_cost_per_sample_usd: Optional[float] = None
    gpu_memory_utilization: float = 0.25
    max_model_len: int = 2048


class ARTUnavailableError(RuntimeError):
    pass


def resolve_art_action_set(action_set_name: str) -> tuple[BudgetPrimitive, ...]:
    if action_set_name != "tool_friendly_v1":
        raise ValueError(
            f"Unsupported DBBench ART action set: {action_set_name}. "
            'Only "tool_friendly_v1" is implemented in the first version.'
        )
    return TOOL_FRIENDLY_BUDGET_PRIMITIVES


def build_dbbench_art_scenario(
    sample_index: SampleIndex,
    dataset_item: DBBenchDatasetItem,
    allowed_primitives: Sequence[BudgetPrimitive],
) -> DBBenchARTScenario:
    try:
        sample_index_int: Optional[int] = int(sample_index)
    except (TypeError, ValueError):
        sample_index_int = None
    sql_type = None
    ground_truth_sql = dataset_item.answer_info.ground_truth_sql
    if ground_truth_sql:
        sql_type = ground_truth_sql.split()[0].upper()
    return DBBenchARTScenario(
        sample_index=sample_index,
        sample_index_int=sample_index_int,
        instruction=dataset_item.instruction,
        table_name=dataset_item.table_info.name,
        column_names=[column.name for column in dataset_item.table_info.column_info_list],
        row_count=len(dataset_item.table_info.row_list),
        column_count=len(dataset_item.table_info.column_info_list),
        skill_list=list(dataset_item.skill_list),
        allowed_actions=[primitive.name for primitive in allowed_primitives],
        ground_truth_sql_type=sql_type,
        metadata={
            "database_name": dataset_item.database_name,
        },
    )


def render_allocator_prompt(scenario: DBBenchARTScenario) -> str:
    column_list = ", ".join(scenario.column_names)
    skill_list = ", ".join(scenario.skill_list)
    allowed_actions = ", ".join(scenario.allowed_actions)
    return (
        f"Sample ID: {scenario.sample_index}\n"
        f"Instruction: {scenario.instruction}\n"
        f"Table: {scenario.table_name}\n"
        f"Columns: {column_list}\n"
        f"Row count: {scenario.row_count}\n"
        f"Column count: {scenario.column_count}\n"
        f"Skills: {skill_list}\n"
        f"Allowed actions: {allowed_actions}\n"
        'Return JSON only in the exact form {"action": "<allowed_action>"}'
    )


def _extract_json_object(raw_text: str) -> Optional[str]:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return raw_text[start : end + 1]


def parse_allocator_output(
    raw_output: str,
    allowed_actions: Sequence[str],
    fallback_action: str,
    strict_json: bool,
) -> tuple[str, dict[str, Any]]:
    parse_failure = False
    payload_str = raw_output if not strict_json else _extract_json_object(raw_output) or ""
    payload: dict[str, Any] = {}
    if payload_str:
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            parse_failure = True
    else:
        parse_failure = True
    action_name = payload.get("action") if isinstance(payload, dict) else None
    if not isinstance(action_name, str) or action_name not in allowed_actions:
        parse_failure = True
        action_name = fallback_action
    return action_name, {
        "parse_failure": parse_failure,
        "raw_output": raw_output,
    }


def _art_training_paths(config: DBBenchARTConfig) -> tuple[Path, Path, Path, Path]:
    artifact_dir = Path(config.output_dir) / "dbbench_art_lora"
    backend_path = Path(config.local_backend_path or (artifact_dir / "backend"))
    manifest_path = artifact_dir / "training_manifest.json"
    rollouts_path = artifact_dir / "grouped_rollouts.jsonl"
    return artifact_dir, backend_path, manifest_path, rollouts_path


def _load_art_modules() -> Any:
    try:
        import art  # type: ignore[import-not-found]
        from art.local import LocalBackend  # type: ignore[import-not-found]
    except ModuleNotFoundError as e:
        raise ARTUnavailableError(
            "ART is not installed in the active environment. Install `openpipe-art` "
            "into /work/conda-py311 before running the DBBench ART path."
        ) from e
    return art, LocalBackend


def _art_internal_model_config(config: DBBenchARTConfig) -> dict[str, Any]:
    return {
        "engine_args": {
            "gpu_memory_utilization": float(config.gpu_memory_utilization),
            "max_model_len": int(config.max_model_len),
        },
        "peft_args": {
            "r": int(config.lora_rank),
            "lora_alpha": int(config.lora_alpha),
        },
    }


def create_art_backend(config: DBBenchARTConfig) -> Any:
    if config.backend != "local":
        raise ValueError(
            f"Unsupported DBBench ART backend: {config.backend}. "
            'Only "local" is implemented in the first version.'
        )
    _, LocalBackend = _load_art_modules()
    _, backend_path, _, _ = _art_training_paths(config)
    backend_path.mkdir(parents=True, exist_ok=True)
    return LocalBackend(path=str(backend_path))


class DBBenchARTPolicy:
    def __init__(
        self,
        config: DBBenchARTConfig,
        adapter_path: str,
        cost_tracker: Optional[CostTracker] = None,
        fallback_action_name: str = ART_FALLBACK_ACTION,
    ) -> None:
        self.config = config
        self.adapter_path = adapter_path
        self.cost_tracker = cost_tracker
        self.fallback_action_name = fallback_action_name
        self._primitive_by_name = {
            primitive.name: primitive for primitive in config.allowed_primitives
        }
        if self.fallback_action_name not in self._primitive_by_name:
            self.fallback_action_name = config.allowed_primitives[0].name
        self.last_selection_meta: dict[str, Any] = {}
        self._manifest_payload: Optional[dict[str, Any]] = None
        self._use_manifest_fallback = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._art = None
        self._backend = None
        self._model = None
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"DBBench ART manifest path does not exist: {adapter_path}"
            )
        with open(adapter_path, "r") as f:
            self._manifest_payload = json.load(f)
        if self._manifest_payload.get("policy_mode") == "manifest_fallback":
            if not self.config.use_manifest_fallback:
                raise RuntimeError(
                    "The provided DBBench ART manifest is a manifest-fallback policy, "
                    "but use_manifest_fallback is disabled."
                )
            self._use_manifest_fallback = True
        else:
            self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._initialize_runtime_async())

    async def _initialize_runtime_async(self) -> None:
        assert self._manifest_payload is not None
        art, LocalBackend = _load_art_modules()
        backend_path = str(
            self._manifest_payload.get(
                "backend_path",
                str(_art_training_paths(self.config)[1]),
            )
        )
        internal_config = self._manifest_payload.get("internal_config") or _art_internal_model_config(
            self.config
        )
        self._backend = LocalBackend(path=backend_path)
        self._model = art.TrainableModel(
            name=str(self._manifest_payload["art_model_name"]),
            project=str(self._manifest_payload.get("project", ART_PROJECT)),
            base_model=str(
                self._manifest_payload.get(
                    "allocator_base_model",
                    self._manifest_payload.get("base_model", self.config.base_model),
                )
            ),
            _internal_config=internal_config,
        )
        await self._model.register(self._backend)
        self._art = art

    def _select_from_manifest(
        self, scenario: DBBenchARTScenario
    ) -> tuple[str, dict[str, Any]]:
        assert self._manifest_payload is not None
        per_skill = self._manifest_payload.get("per_skill_best_action", {})
        action_scores = {
            action_name: 0.0 for action_name in self._primitive_by_name.keys()
        }
        skill_hits = 0
        for skill in scenario.skill_list:
            if skill in per_skill:
                skill_hits += 1
                best_action = per_skill[skill]
                if best_action in action_scores:
                    action_scores[best_action] += 1.0
        global_best = self._manifest_payload.get(
            "global_best_action", self.fallback_action_name
        )
        chosen_action = max(
            scenario.allowed_actions,
            key=lambda action_name: (
                float(action_scores.get(action_name, 0.0)),
                1.0 if action_name == global_best else 0.0,
                -float(self._primitive_by_name[action_name].static_cost_proxy),
            ),
        )
        return chosen_action, {
            "policy_mode": "manifest_fallback",
            "skill_hits": skill_hits,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "parse_failure": False,
        }

    async def _select_with_art_async(
        self, scenario: DBBenchARTScenario
    ) -> tuple[str, dict[str, Any]]:
        assert self._model is not None
        prompt = render_allocator_prompt(scenario)
        messages = [
            {"role": "system", "content": ART_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = await self._model.openai_client().chat.completions.create(
            messages=messages,
            model=self._model.get_inference_name(),
            max_tokens=int(self.config.max_allocator_tokens),
            temperature=max(0.05, float(self.config.temperature)),
            n=1,
        )
        choice = response.choices[0]
        raw_output = (choice.message.content or "").strip()
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        action_name, parse_meta = parse_allocator_output(
            raw_output=raw_output,
            allowed_actions=scenario.allowed_actions,
            fallback_action=self.fallback_action_name,
            strict_json=bool(self.config.strict_json),
        )
        return action_name, {
            "policy_mode": "art_local_backend",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            **parse_meta,
        }

    def select_action(self, scenario: DBBenchARTScenario) -> str:
        if self._use_manifest_fallback:
            action_name, selection_meta = self._select_from_manifest(scenario)
        else:
            if self._loop is None:
                raise RuntimeError("DBBench ART runtime loop is not initialized.")
            action_name, selection_meta = self._loop.run_until_complete(
                self._select_with_art_async(scenario)
            )
        self.last_selection_meta = selection_meta
        return action_name

    def select_primitive(self, scenario: DBBenchARTScenario) -> BudgetPrimitive:
        action_name = self.select_action(scenario)
        return self._primitive_by_name.get(
            action_name, self._primitive_by_name[self.fallback_action_name]
        )

    @property
    def manifest_payload(self) -> dict[str, Any]:
        return dict(self._manifest_payload or {})


class DBBenchARTTrainer:
    def __init__(self, config: DBBenchARTConfig):
        self.config = config
        self._primitive_by_name = {
            primitive.name: primitive for primitive in config.allowed_primitives
        }

    @staticmethod
    def _get_dataset_item(task: Any, sample_index: SampleIndex) -> DBBenchDatasetItem:
        dataset = getattr(task, "_Task__dataset", None)
        if isinstance(dataset, dict) and sample_index in dataset:
            return dataset[sample_index]
        getter = getattr(task, "_Task__get_dataset_item", None)
        if callable(getter):
            return getter(sample_index)
        raise RuntimeError(
            "Unable to access DBBench dataset items for ART scenario construction."
        )

    def build_scenarios(
        self,
        task: Any,
        sample_order: Optional[Sequence[SampleIndex]] = None,
    ) -> list[DBBenchARTScenario]:
        if sample_order is None:
            sample_order = task.get_sample_index_list()
        scenario_list: list[DBBenchARTScenario] = []
        for sample_index in sample_order:
            dataset_item = self._get_dataset_item(task, sample_index)
            scenario_list.append(
                build_dbbench_art_scenario(
                    sample_index=sample_index,
                    dataset_item=dataset_item,
                    allowed_primitives=self.config.allowed_primitives,
                )
            )
        return scenario_list

    def _run_single_rollout(
        self,
        task: Any,
        agent: Any,
        scenario: DBBenchARTScenario,
        primitive: BudgetPrimitive,
        cost_tracker: Optional[CostTracker],
    ) -> DBBenchARTRolloutResult:
        calls = getattr(cost_tracker, "calls", None) or []
        start_idx = len(calls)
        session = Session(task_name=task.task_name, sample_index=scenario.sample_index)
        task.reset(session)
        try:
            set_generation_token_budget(agent, primitive.token_budget)
            task.max_round = primitive.max_round
            if hasattr(task, "tool_budget"):
                setattr(task, "tool_budget", int(primitive.tool_budget))
            if hasattr(task, "stop_enabled"):
                setattr(task, "stop_enabled", bool(primitive.stop_enabled))
            while session.sample_status == SampleStatus.RUNNING:
                agent.inference(session)
                if session.sample_status != SampleStatus.RUNNING:
                    break
                task.interact(session)
        except Exception as exc:
            if session.sample_status == SampleStatus.RUNNING:
                session.sample_status = SampleStatus.TASK_UNKNOWN_ERROR
                if session.finish_reason is None:
                    session.finish_reason = f"ART rollout failed: {exc}"
            raise
        finally:
            if (
                getattr(task, "current_sample_index", None) == scenario.sample_index
                and session.sample_status != SampleStatus.INITIAL
            ):
                if session.sample_status == SampleStatus.RUNNING:
                    session.sample_status = SampleStatus.TASK_UNKNOWN_ERROR
                    if session.finish_reason is None:
                        session.finish_reason = (
                            "ART rollout exited before task completion."
                        )
                task.complete(session)
        calls = getattr(cost_tracker, "calls", None) or []
        end_idx = len(calls)
        sample_calls = calls[start_idx:end_idx]
        prompt_tokens = int(sum(c.prompt_tokens for c in sample_calls))
        completion_tokens = int(sum(c.completion_tokens for c in sample_calls))
        cost_usd = float(sum(c.total_cost_usd for c in sample_calls))
        cost_normalized = cost_usd / max(self.config.budget_target_usd, 1e-9)
        correct = session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
        reward = (1.0 if correct else 0.0) - (
            float(self.config.reward_lambda) * cost_normalized
        )
        return DBBenchARTRolloutResult(
            sample_index=scenario.sample_index,
            action_name=primitive.name,
            correct=bool(correct),
            sample_status=session.sample_status.value,
            cost_usd=cost_usd,
            cost_normalized=cost_normalized,
            reward=reward,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={
                "evaluation_outcome": session.evaluation_record.outcome.value,
            },
        )

    def run_group_rollout(
        self,
        task: Any,
        agent: Any,
        scenario: DBBenchARTScenario,
        k: int,
        cost_tracker: Optional[CostTracker] = None,
    ) -> list[DBBenchARTRolloutResult]:
        k = max(1, min(int(k), int(self.config.max_rollouts_per_scenario)))
        rollout_list: list[DBBenchARTRolloutResult] = []
        action_names = list(scenario.allowed_actions)
        for idx in range(k):
            action_name = action_names[idx % len(action_names)]
            rollout_list.append(
                self._run_single_rollout(
                    task=task,
                    agent=agent,
                    scenario=scenario,
                    primitive=self._primitive_by_name[action_name],
                    cost_tracker=cost_tracker,
                )
            )
        return rollout_list

    def _append_group_rollout_log(
        self,
        rollouts_path: Path,
        scenario: DBBenchARTScenario,
        rollout_list: Sequence[DBBenchARTRolloutResult],
    ) -> None:
        with open(rollouts_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "scenario": asdict(scenario),
                        "rollouts": [asdict(rollout) for rollout in rollout_list],
                    }
                )
                + "\n"
            )

    async def _build_art_trajectory_group(
        self,
        art_module: Any,
        model: Any,
        task: Any,
        agent: Any,
        scenario: DBBenchARTScenario,
        cost_tracker: Optional[CostTracker],
        rollouts_path: Path,
    ) -> Any:
        messages = [
            {"role": "system", "content": ART_SYSTEM_PROMPT},
            {"role": "user", "content": render_allocator_prompt(scenario)},
        ]
        response = await model.openai_client().chat.completions.create(
            messages=messages,
            model=model.get_inference_name(),
            max_tokens=int(self.config.max_allocator_tokens),
            temperature=max(0.05, float(self.config.temperature)),
            n=max(1, int(self.config.group_size)),
        )
        usage = getattr(response, "usage", None)
        group_prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        group_completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        choice_count = max(1, len(response.choices))
        per_choice_completion_tokens = int(group_completion_tokens / choice_count)

        trajectory_list = []
        rollout_results: list[DBBenchARTRolloutResult] = []
        for choice in response.choices:
            raw_output = (choice.message.content or "").strip()
            action_name, parse_meta = parse_allocator_output(
                raw_output=raw_output,
                allowed_actions=scenario.allowed_actions,
                fallback_action=ART_FALLBACK_ACTION,
                strict_json=bool(self.config.strict_json),
            )
            primitive = self._primitive_by_name.get(
                action_name,
                self._primitive_by_name[ART_FALLBACK_ACTION],
            )
            rollout = self._run_single_rollout(
                task=task,
                agent=agent,
                scenario=scenario,
                primitive=primitive,
                cost_tracker=cost_tracker,
            )
            rollout_results.append(rollout)
            trajectory_list.append(
                art_module.Trajectory(
                    messages_and_choices=[*messages, choice],
                    reward=float(rollout.reward),
                    metrics={
                        "cost_usd": float(rollout.cost_usd),
                        "cost_normalized": float(rollout.cost_normalized),
                        "solver_prompt_tokens": int(rollout.prompt_tokens),
                        "solver_completion_tokens": int(rollout.completion_tokens),
                        "allocator_prompt_tokens": group_prompt_tokens,
                        "allocator_completion_tokens": per_choice_completion_tokens,
                        "correct": 1 if rollout.correct else 0,
                    },
                    metadata={
                        "sample_index": str(scenario.sample_index),
                        "chosen_action": str(primitive.name),
                        "sample_status": str(rollout.sample_status),
                        "solver_correct": bool(rollout.correct),
                        "parse_failure": bool(parse_meta["parse_failure"]),
                    },
                    logs=[f"allocator_output={raw_output}"],
                )
            )
        self._append_group_rollout_log(rollouts_path, scenario, rollout_results)
        mean_reward = (
            sum(float(rollout.reward) for rollout in rollout_results) / len(rollout_results)
            if rollout_results
            else 0.0
        )
        return art_module.TrajectoryGroup(
            trajectory_list,
            metadata={"sample_index": str(scenario.sample_index)},
            metrics={"mean_reward": mean_reward},
        )

    def _make_model_name(self) -> str:
        return f"dbbench-art-allocator-{int(time.time())}"

    async def train_async(
        self,
        task: Any,
        agent: Any,
        sample_order: Optional[Sequence[SampleIndex]] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> str:
        art_module, _ = _load_art_modules()
        artifact_dir, backend_path, manifest_path, rollouts_path = _art_training_paths(self.config)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        backend_path.mkdir(parents=True, exist_ok=True)
        scenario_list = self.build_scenarios(task=task, sample_order=sample_order)
        if not scenario_list:
            raise RuntimeError("No DBBench scenarios available for ART training.")

        backend = create_art_backend(self.config)
        model = art_module.TrainableModel(
            name=self._make_model_name(),
            project=ART_PROJECT,
            base_model=self.config.base_model,
            _internal_config=_art_internal_model_config(self.config),
        )
        await model.register(backend)

        latest_train_result = None
        cost_limit = 1.25 * float(
            self.config.baseline_mean_cost_per_sample_usd
            if self.config.baseline_mean_cost_per_sample_usd is not None
            else self.config.budget_target_usd
        )
        running_cost_sum = 0.0
        running_rollouts = 0
        early_stop_reason: Optional[str] = None
        skipped_equal_reward_groups = 0

        for epoch_idx in range(int(self.config.max_training_epochs)):
            for scenario in scenario_list:
                group = await self._build_art_trajectory_group(
                    art_module=art_module,
                    model=model,
                    task=task,
                    agent=agent,
                    scenario=scenario,
                    cost_tracker=cost_tracker,
                    rollouts_path=rollouts_path,
                )
                finished_groups = await art_module.gather_trajectory_groups(
                    [asyncio.sleep(0, result=group)],
                    pbar_desc=f"dbbench-art-e{epoch_idx + 1}",
                )
                await model.log(finished_groups, split="train")
                reward_values = [
                    float(getattr(trajectory, "reward", 0.0))
                    for finished_group in finished_groups
                    for trajectory in finished_group
                ]
                if reward_values and (max(reward_values) - min(reward_values) <= 1e-9):
                    skipped_equal_reward_groups += 1
                    await model.log(
                        metrics={
                            "skipped_equal_reward_group": 1.0,
                            "group_reward": reward_values[0],
                        },
                        split="train",
                    )
                else:
                    latest_train_result = await backend.train(
                        model,
                        finished_groups,
                        learning_rate=float(self.config.art_learning_rate),
                    )
                    await model.log(
                        metrics=latest_train_result.metrics,
                        split="train",
                        step=latest_train_result.step,
                    )
                for finished_group in finished_groups:
                    for trajectory in finished_group:
                        cost_value = float(trajectory.metrics.get("cost_usd", 0.0))
                        running_cost_sum += cost_value
                        running_rollouts += 1
                running_mean_cost = (
                    running_cost_sum / running_rollouts if running_rollouts > 0 else 0.0
                )
                if running_rollouts > 0 and running_mean_cost > cost_limit:
                    early_stop_reason = (
                        "mean_rollout_cost_exceeded_limit"
                    )
                    break
            if early_stop_reason is not None:
                break

        try:
            await model.delete_checkpoints()
        except Exception:
            pass

        manifest_payload = {
            "policy_version": ART_POLICY_VERSION,
            "policy_mode": "art_local_backend",
            "art_model_name": str(model.name),
            "project": ART_PROJECT,
            "allocator_base_model": str(self.config.base_model),
            "base_model": str(self.config.base_model),
            "backend_path": str(backend_path),
            "reward_lambda": float(self.config.reward_lambda),
            "budget_target_usd": float(self.config.budget_target_usd),
            "group_size": int(self.config.group_size),
            "temperature": float(self.config.temperature),
            "epochs": int(self.config.max_training_epochs),
            "training_scenarios": int(len(scenario_list)),
            "groups_per_step": int(self.config.groups_per_step),
            "lora_rank": int(self.config.lora_rank),
            "lora_alpha": int(self.config.lora_alpha),
            "action_set": [primitive.name for primitive in self.config.allowed_primitives],
            "internal_config": _art_internal_model_config(self.config),
            "latest_step": int(getattr(latest_train_result, "step", 0) or 0),
            "checkpoint_path": (
                None
                if latest_train_result is None
                else getattr(latest_train_result, "checkpoint_path", None)
            ),
            "early_stop_reason": early_stop_reason,
            "skipped_equal_reward_groups": int(skipped_equal_reward_groups),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest_payload, f, indent=2)
        return str(manifest_path)

    def train(
        self,
        task: Any,
        agent: Any,
        sample_order: Optional[Sequence[SampleIndex]] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> str:
        return asyncio.run(
            self.train_async(
                task=task,
                agent=agent,
                sample_order=sample_order,
                cost_tracker=cost_tracker,
            )
        )
