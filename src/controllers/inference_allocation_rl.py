from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import os
import random
import re
from typing import Any, Mapping, Optional, Sequence

from src.callbacks import CallbackArguments, CallbackHandler
from src.tasks import Task, DatasetItem
from src.agents import Agent
from src.utils import set_generation_token_budget
from src.typings import (
    ChatHistory,
    ChatHistoryItem,
    Role,
    SampleStatus,
    Session,
    SessionEvaluationOutcome,
)

try:  # Keep module importable in lightweight local environments.
    import torch
    from torch import nn
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


OS_RL_ACTIONS: tuple[str, ...] = (
    "retrieve_none",
    "retrieve_top1",
    "retrieve_top4",
    "set_budget_small",
    "set_budget_medium",
    "set_budget_large",
    "reason_step",
    "verify",
    "retry",
    "terminate",
)

OS_RL_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "instruction_chars_1k",
    "skill_count_10",
    "stage_setup",
    "stage_running",
    "stage_terminal",
    "attempt_fraction",
    "decision_fraction",
    "round_fraction",
    "token_budget_1536",
    "max_round_10",
    "tool_budget_16",
    "replay_none",
    "replay_top1",
    "replay_top4",
    "retrieval_injection_fraction",
    "verification_fraction",
    "verifier_score",
    "verifier_uncertainty",
    "cost_normalized",
    "running_accuracy",
    "recent_task_limit_rate",
    "previous_sample_cost_normalized",
    "previous_sample_correct",
    "status_initial",
    "status_running",
    "status_completed",
    "status_task_limit",
    "status_validation_failed",
    "status_other_error",
    "retrieved_example_count_4",
    "top1_similarity_score",
    "avg_similarity_score",
    "avg_rounds_in_retrieved_10",
    "avg_cost_in_retrieved_normalized",
)


@dataclass(frozen=True)
class RLBudgetProfile:
    name: str
    token_budget: int
    max_round: int
    tool_budget: int
    stop_enabled: bool = True


OS_RL_BUDGET_PROFILES: dict[str, RLBudgetProfile] = {
    "small": RLBudgetProfile(
        name="small",
        token_budget=512,
        max_round=3,
        tool_budget=6,
    ),
    "medium": RLBudgetProfile(
        name="medium",
        token_budget=768,
        max_round=3,
        tool_budget=6,
    ),
    "large": RLBudgetProfile(
        name="large",
        token_budget=1024,
        max_round=4,
        tool_budget=8,
    ),
}


@dataclass
class MaskedDQNConfig:
    state_dim: int
    action_names: tuple[str, ...] = OS_RL_ACTIONS
    gamma: float = 0.95
    lr: float = 1e-3
    epsilon_start: float = 0.25
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_interval: int = 50
    replay_buffer_size: int = 4096
    batch_size: int = 32
    hidden_dim: int = 96
    seed: int = 42


@dataclass
class RLEpisodeTransition:
    state: list[float]
    action_idx: int
    next_state: list[float]
    next_valid_action_mask: list[bool]
    done: bool


@dataclass
class RLEpisodeResult:
    session: Session
    callback_args: CallbackArguments
    summary: dict[str, Any]


@dataclass
class _EpisodeState:
    stage: str = "setup"
    attempt_index: int = 0
    decision_step: int = 0
    replay_mode: str = "none"
    replay_sample_count: Optional[int] = None
    budget_name: str = "medium"
    setup_retrieval_selected: bool = False
    setup_budget_selected: bool = False
    force_reason_next: bool = False
    reason_steps_taken: int = 0
    retrieval_injections: int = 0
    verification_count: int = 0
    verifier_score: float = 0.5
    verifier_uncertainty: float = 1.0
    retry_count: int = 0
    forced_terminate: bool = False
    last_status: str = "initial"
    cumulative_cost_usd: float = 0.0


class _QNetwork(nn.Module if nn is not None else object):  # type: ignore[misc,valid-type]
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        if nn is None:
            raise RuntimeError("PyTorch is required for MaskedDQNInferenceController.")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)


class MaskedDQNInferenceController:
    def __init__(self, config: MaskedDQNConfig) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for --enable_inference_rl. "
                "Install torch or run from the LAB virtual environment."
            )
        self.config = config
        self.action_names = tuple(config.action_names)
        self.rng = random.Random(config.seed)
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            torch.manual_seed(config.seed)
            self.policy_net = _QNetwork(
                config.state_dim, len(self.action_names), config.hidden_dim
            )
            self.target_net = _QNetwork(
                config.state_dim, len(self.action_names), config.hidden_dim
            )
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.replay_buffer = deque(maxlen=int(config.replay_buffer_size))
        self.training_steps = 0
        self.episodes_seen = 0
        self.epsilon = float(config.epsilon_start)

    def select_action(
        self, state: Sequence[float], valid_action_mask: Sequence[bool]
    ) -> tuple[int, dict[str, Any]]:
        valid_indices = [
            idx for idx, is_valid in enumerate(valid_action_mask) if bool(is_valid)
        ]
        if not valid_indices:
            terminate_idx = self.action_names.index("terminate")
            valid_indices = [terminate_idx]
        if self.rng.random() < self.epsilon:
            action_idx = self.rng.choice(valid_indices)
            return action_idx, {
                "epsilon": float(self.epsilon),
                "explore": True,
                "q_values": None,
            }
        with torch.no_grad():
            x = torch.tensor([list(state)], dtype=torch.float32)
            q_values = self.policy_net(x)[0]
            masked_q = q_values.clone()
            for idx, is_valid in enumerate(valid_action_mask):
                if not bool(is_valid):
                    masked_q[idx] = -1.0e9
            action_idx = int(torch.argmax(masked_q).item())
            q_list = [float(value) for value in q_values.detach().cpu().tolist()]
        return action_idx, {
            "epsilon": float(self.epsilon),
            "explore": False,
            "q_values": q_list,
        }

    def observe_episode(
        self, transitions: Sequence[RLEpisodeTransition], final_reward: float
    ) -> dict[str, Any]:
        if not transitions:
            return {"updates": 0, "loss": None, "buffer_size": len(self.replay_buffer)}
        for idx, transition in enumerate(transitions):
            done = idx == len(transitions) - 1
            reward = float(final_reward) if done else 0.0
            self._append_transition(
                (
                    list(transition.state),
                    int(transition.action_idx),
                    reward,
                    list(transition.next_state),
                    list(transition.next_valid_action_mask),
                    bool(done or transition.done),
                )
            )
        losses = []
        updates = max(1, min(4, len(transitions)))
        for _ in range(updates):
            loss = self._train_step()
            if loss is not None:
                losses.append(loss)
        self.episodes_seen += 1
        self.epsilon = max(
            float(self.config.epsilon_end),
            float(self.epsilon) * float(self.config.epsilon_decay),
        )
        return {
            "updates": len(losses),
            "loss": (sum(losses) / len(losses)) if losses else None,
            "buffer_size": len(self.replay_buffer),
        }

    def _append_transition(
        self,
        transition: tuple[list[float], int, float, list[float], list[bool], bool],
    ) -> None:
        self.replay_buffer.append(transition)

    def _train_step(self) -> Optional[float]:
        if len(self.replay_buffer) == 0:
            return None
        batch_size = min(int(self.config.batch_size), len(self.replay_buffer))
        batch = self.rng.sample(list(self.replay_buffer), batch_size)
        states, actions, rewards, next_states, next_masks, dones = zip(*batch)
        state_tensor = torch.tensor(states, dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
        done_tensor = torch.tensor(dones, dtype=torch.float32)
        q_values = self.policy_net(state_tensor).gather(1, action_tensor).squeeze(1)
        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_tensor)
            next_mask_tensor = torch.tensor(next_masks, dtype=torch.bool)
            next_q_policy = next_q_policy.masked_fill(~next_mask_tensor, -1.0e9)
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)
            next_q_target = self.target_net(next_state_tensor)
            next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
            target = reward_tensor + (
                (1.0 - done_tensor) * float(self.config.gamma) * next_q_values
            )
        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        self.training_steps += 1
        if self.training_steps % int(self.config.target_update_interval) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.detach().cpu().item())

    def save_state(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "replay_buffer": list(self.replay_buffer),
                "training_steps": self.training_steps,
                "episodes_seen": self.episodes_seen,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load_state(self, path: str) -> None:
        if not os.path.exists(path):
            return
        payload = torch.load(path, map_location="cpu")
        saved_config = payload.get("config", {})
        saved_action_names = tuple(saved_config.get("action_names", ()))
        if int(saved_config.get("state_dim", -1)) != int(self.config.state_dim):
            raise RuntimeError(
                "RL state file is incompatible with current config: "
                f"state_dim={saved_config.get('state_dim')} saved, "
                f"{self.config.state_dim} requested."
            )
        if saved_action_names != tuple(self.config.action_names):
            raise RuntimeError(
                "RL state file is incompatible with current config: "
                f"action_names={saved_action_names} saved, "
                f"{tuple(self.config.action_names)} requested."
            )
        if int(saved_config.get("hidden_dim", -1)) != int(self.config.hidden_dim):
            raise RuntimeError(
                "RL state file is incompatible with current config: "
                f"hidden_dim={saved_config.get('hidden_dim')} saved, "
                f"{self.config.hidden_dim} requested."
            )
        self.policy_net.load_state_dict(payload["policy_net"])
        self.target_net.load_state_dict(payload["target_net"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.replay_buffer = deque(
            payload.get("replay_buffer", []),
            maxlen=int(self.config.replay_buffer_size),
        )
        self.training_steps = int(payload.get("training_steps", 0))
        self.episodes_seen = int(payload.get("episodes_seen", 0))
        self.epsilon = float(payload.get("epsilon", self.config.epsilon_start))


def apply_os_rl_budget_profile(agent: Agent, task: Task[Any], budget_name: str) -> None:
    profile = OS_RL_BUDGET_PROFILES[budget_name]
    set_generation_token_budget(agent, profile.token_budget)
    task.max_round = profile.max_round
    if hasattr(task, "tool_budget"):
        setattr(task, "tool_budget", int(profile.tool_budget))
    if hasattr(task, "stop_enabled"):
        setattr(task, "stop_enabled", bool(profile.stop_enabled))


def compute_final_reward(
    *,
    correct: bool,
    total_cost_usd: float,
    budget_target_usd: float,
    cost_lambda: float,
    validation_failed: bool,
    retry_count: int,
    forced_terminate: bool,
) -> float:
    normalized_cost = float(total_cost_usd) / max(float(budget_target_usd), 1e-9)
    reward = (1.0 if correct else 0.0) - (float(cost_lambda) * normalized_cost)
    if validation_failed:
        reward -= 0.10
    reward -= 0.03 * float(max(0, retry_count))
    if forced_terminate:
        reward -= 0.20
    return float(reward)


def build_os_rl_state(
    *,
    dataset_item: DatasetItem,
    episode_state: _EpisodeState,
    current_round: int,
    max_round: int,
    max_decisions: int,
    max_attempts: int,
    max_retrieval_injections: int,
    max_verifications: int,
    retrieval_summary: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
    budget_target_usd: float,
) -> tuple[list[float], dict[str, float]]:
    instruction = str(getattr(dataset_item, "instruction", "") or "")
    skill_count = len(getattr(dataset_item, "skill_list", []) or [])
    budget_profile = OS_RL_BUDGET_PROFILES[episode_state.budget_name]
    status = episode_state.last_status
    values = {
        "bias": 1.0,
        "instruction_chars_1k": min(len(instruction) / 1000.0, 5.0),
        "skill_count_10": min(skill_count / 10.0, 5.0),
        "stage_setup": 1.0 if episode_state.stage == "setup" else 0.0,
        "stage_running": 1.0 if episode_state.stage == "running" else 0.0,
        "stage_terminal": 1.0 if episode_state.stage == "terminal" else 0.0,
        "attempt_fraction": float(episode_state.attempt_index)
        / float(max(max_attempts - 1, 1)),
        "decision_fraction": float(episode_state.decision_step)
        / float(max(max_decisions, 1)),
        "round_fraction": float(current_round) / float(max(max_round, 1)),
        "token_budget_1536": float(budget_profile.token_budget) / 1536.0,
        "max_round_10": float(budget_profile.max_round) / 10.0,
        "tool_budget_16": float(budget_profile.tool_budget) / 16.0,
        "replay_none": 1.0 if episode_state.replay_mode == "none" else 0.0,
        "replay_top1": (
            1.0
            if episode_state.replay_mode == "retrieved"
            and episode_state.replay_sample_count == 1
            else 0.0
        ),
        "replay_top4": (
            1.0
            if episode_state.replay_mode == "retrieved"
            and episode_state.replay_sample_count == 4
            else 0.0
        ),
        "retrieval_injection_fraction": float(episode_state.retrieval_injections)
        / float(max(max_retrieval_injections, 1)),
        "verification_fraction": float(episode_state.verification_count)
        / float(max(max_verifications, 1)),
        "verifier_score": float(episode_state.verifier_score),
        "verifier_uncertainty": float(episode_state.verifier_uncertainty),
        "cost_normalized": float(episode_state.cumulative_cost_usd)
        / max(float(budget_target_usd), 1e-9),
        "running_accuracy": float(runtime_state.get("running_accuracy", 0.0) or 0.0),
        "recent_task_limit_rate": float(
            runtime_state.get("recent_task_limit_rate", 0.0) or 0.0
        ),
        "previous_sample_cost_normalized": float(
            runtime_state.get("previous_sample_cost_normalized", 0.0) or 0.0
        ),
        "previous_sample_correct": (
            1.0 if bool(runtime_state.get("previous_sample_correct", False)) else 0.0
        ),
        "status_initial": 1.0 if status == "initial" else 0.0,
        "status_running": 1.0 if status == SampleStatus.RUNNING.value else 0.0,
        "status_completed": 1.0 if status == SampleStatus.COMPLETED.value else 0.0,
        "status_task_limit": (
            1.0 if status == SampleStatus.TASK_LIMIT_REACHED.value else 0.0
        ),
        "status_validation_failed": (
            1.0 if status == SampleStatus.AGENT_VALIDATION_FAILED.value else 0.0
        ),
        "status_other_error": (
            1.0
            if status
            not in {
                "initial",
                SampleStatus.RUNNING.value,
                SampleStatus.COMPLETED.value,
                SampleStatus.TASK_LIMIT_REACHED.value,
                SampleStatus.AGENT_VALIDATION_FAILED.value,
            }
            else 0.0
        ),
        "retrieved_example_count_4": min(
            float(retrieval_summary.get("retrieved_example_count", 0.0) or 0.0) / 4.0,
            1.0,
        ),
        "top1_similarity_score": float(
            retrieval_summary.get("top1_similarity_score", 0.0) or 0.0
        ),
        "avg_similarity_score": float(
            retrieval_summary.get("avg_similarity_score", 0.0) or 0.0
        ),
        "avg_rounds_in_retrieved_10": float(
            retrieval_summary.get("avg_rounds_in_retrieved", 0.0) or 0.0
        )
        / 10.0,
        "avg_cost_in_retrieved_normalized": float(
            retrieval_summary.get("avg_cost_in_retrieved_normalized", 0.0) or 0.0
        ),
    }
    return [float(values[name]) for name in OS_RL_FEATURE_NAMES], values


def valid_os_rl_action_mask(
    episode_state: _EpisodeState,
    *,
    current_session: Optional[Session],
    max_attempts: int,
    max_retrieval_injections: int,
    max_verifications: int,
) -> list[bool]:
    stage = episode_state.stage
    mask = {name: False for name in OS_RL_ACTIONS}
    if episode_state.force_reason_next and stage in {"setup", "running"}:
        mask["reason_step"] = True
        return [bool(mask[name]) for name in OS_RL_ACTIONS]
    if stage == "setup":
        can_select_retrieval = (
            not episode_state.setup_retrieval_selected
            and episode_state.retrieval_injections < max_retrieval_injections
        )
        mask["retrieve_top1"] = (
            can_select_retrieval and episode_state.replay_sample_count != 1
        )
        # Bootstrap curriculum: keep top-4 and mid-task memory out of the
        # initial online RL problem until the controller learns stable progress.
        mask["retrieve_top4"] = False
        can_select_budget = not episode_state.setup_budget_selected
        mask["set_budget_small"] = (
            can_select_budget and episode_state.budget_name != "small"
        )
        mask["set_budget_medium"] = (
            can_select_budget and episode_state.budget_name != "medium"
        )
        mask["set_budget_large"] = (
            can_select_budget and episode_state.budget_name != "large"
        )
        mask["reason_step"] = True
    elif stage == "running":
        # During the first RL curriculum, running attempts should mostly be
        # environment progress. Terminal-state retry/verification handles
        # recovery after the attempt reaches a benchmark status.
        mask["reason_step"] = True
    elif stage == "terminal":
        mask["verify"] = (
            current_session is not None
            and episode_state.verification_count < max_verifications
        )
        mask["retry"] = episode_state.attempt_index + 1 < max_attempts
        mask["terminate"] = True
    return [bool(mask[name]) for name in OS_RL_ACTIONS]


def _get_cost_tracker_calls(cost_tracker: Any) -> list[Any]:
    return list(getattr(cost_tracker, "calls", None) or [])


def _get_call_window_cost_usd(cost_tracker: Any, start_idx: int, end_idx: int) -> float:
    calls = _get_cost_tracker_calls(cost_tracker)
    return float(sum(call.total_cost_usd for call in calls[start_idx:end_idx]))


def _set_task_replay_controls(
    task: Task[Any], replay_mode: str, replay_sample_count: Optional[int]
) -> None:
    task._copal_replay_mode = replay_mode  # type: ignore[attr-defined]
    task._copal_replay_sample_count = replay_sample_count  # type: ignore[attr-defined]


def _abort_current_task_attempt(task: Task[Any]) -> None:
    container = getattr(task, "container", None)
    if container is not None:
        try:
            container.terminate()
        except Exception:  # noqa: BLE001
            pass
        setattr(task, "container", None)
    task.current_sample_index = None
    task.current_round = 0
    if hasattr(task, "_Task__current_dataset_item"):
        setattr(task, "_Task__current_dataset_item", None)


def _render_memory_hint(
    *,
    callback_dict: Mapping[str, Any],
    task: Task[Any],
    agent: Agent,
    sample_index: Any,
    replay_sample_count: int,
) -> tuple[str, dict[str, Any]]:
    selected_sessions: list[Session] = []
    renderer = None
    selected_indices: list[Any] = []
    for callback in callback_dict.values():
        select_retrieved = getattr(callback, "_select_retrieved_sessions", None)
        render_session = getattr(callback, "_render_session_block", None)
        if callable(select_retrieved) and callable(render_session):
            selected_sessions = list(
                select_retrieved(
                    task=task,
                    current_sample_index=sample_index,
                    replay_sample_count=replay_sample_count,
                )
            )
            renderer = render_session
            break
    if not selected_sessions:
        for callback in callback_dict.values():
            utilized_session_list = getattr(callback, "utilized_session_list", None)
            render_session = getattr(callback, "_render_session_block", None)
            if isinstance(utilized_session_list, list) and callable(render_session):
                selected_sessions = list(utilized_session_list[-replay_sample_count:])
                renderer = render_session
                break
    if not selected_sessions or renderer is None:
        return "", {"selected_sample_indices": []}
    agent_role_dict = agent.get_role_dict()
    blocks = []
    for memory_session in selected_sessions:
        selected_indices.append(memory_session.sample_index)
        try:
            blocks.append(renderer(memory_session, agent_role_dict))
        except Exception:  # noqa: BLE001
            continue
    if not blocks:
        return "", {"selected_sample_indices": []}
    hint = (
        "\n\nAdditional retrieved memory for this task. Use only what is relevant, "
        "and continue following the required Act format:\n" + "\n".join(blocks)
    )
    return hint, {"selected_sample_indices": selected_indices}


def append_memory_hint_to_pending_user(
    session: Session,
    *,
    callback_dict: Mapping[str, Any],
    task: Task[Any],
    agent: Agent,
    sample_index: Any,
    replay_sample_count: int,
) -> dict[str, Any]:
    last_item = session.chat_history.get_item_deep_copy(-1)
    if last_item.role != Role.USER:
        return {"injected": False, "reason": "last_message_not_user"}
    hint, metadata = _render_memory_hint(
        callback_dict=callback_dict,
        task=task,
        agent=agent,
        sample_index=sample_index,
        replay_sample_count=replay_sample_count,
    )
    if not hint:
        return {"injected": False, "reason": "no_memory", **metadata}
    session.chat_history.set(
        -1,
        ChatHistoryItem(role=Role.USER, content=str(last_item.content or "") + hint),
    )
    return {"injected": True, "hint_chars": len(hint), **metadata}


def _trajectory_text(session: Session, max_chars: int = 5000) -> str:
    lines = []
    for item_index in range(session.chat_history.get_value_length()):
        item = session.chat_history.get_item_deep_copy(item_index)
        content = " ".join(str(item.content or "").split())
        lines.append(f"{item.role.value}: {content}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_self_critique_verifier(
    *,
    agent: Agent,
    session: Optional[Session],
    verifier_mode: str,
) -> tuple[float, float, dict[str, Any]]:
    if verifier_mode == "none" or session is None:
        return 0.5, 1.0, {"status": "disabled"}
    try:
        language_model = getattr(agent, "_language_model", None)
        if language_model is None:
            return 0.5, 1.0, {"status": "no_language_model"}
        prompt = (
            "You are a verifier for a Linux task-solving agent. "
            "Do not use hidden answers or benchmark evaluators. "
            "Judge only whether the visible trajectory appears likely to have solved "
            "the user's task and followed the required Act format. "
            "Return JSON with keys score, uncertainty, and reason. "
            "score and uncertainty must be numbers from 0 to 1.\n\n"
            f"Trajectory:\n{_trajectory_text(session)}"
        )
        verifier_history = ChatHistory(
            value=[ChatHistoryItem(role=Role.USER, content=prompt)]
        )
        inference_config = (
            {"max_completion_tokens": 160, "temperature": 0.0}
            if hasattr(language_model, "model_name")
            else {"max_new_tokens": 160, "do_sample": False}
        )
        result = language_model.inference(
            [verifier_history],
            inference_config,
            "You are a careful verifier that returns compact JSON.",
        )[0]
        content = str(result.content or "")
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        payload = json.loads(json_match.group(0) if json_match else content)
        score = min(1.0, max(0.0, float(payload.get("score", 0.5))))
        uncertainty = min(1.0, max(0.0, float(payload.get("uncertainty", 0.5))))
        return (
            score,
            uncertainty,
            {
                "status": "ok",
                "raw": content[:1000],
                "reason": str(payload.get("reason", ""))[:500],
            },
        )
    except Exception as e:  # noqa: BLE001
        return 0.5, 1.0, {"status": "failed", "error": str(e)}


def _get_retrieval_feature_summary(
    *,
    callback_dict: Mapping[str, Any],
    task: Task[Any],
    current_sample_index: Any,
    replay_sample_count: Optional[int],
    budget_target_usd: float,
) -> dict[str, Any]:
    for callback in callback_dict.values():
        summary_getter = getattr(callback, "get_retrieval_feature_summary", None)
        if callable(summary_getter):
            summary = dict(
                summary_getter(
                    task=task,
                    current_sample_index=current_sample_index,
                    replay_sample_count=replay_sample_count,
                )
            )
            avg_cost_usd = float(summary.get("avg_cost_in_retrieved_usd", 0.0) or 0.0)
            summary["avg_cost_in_retrieved_normalized"] = avg_cost_usd / max(
                float(budget_target_usd), 1e-9
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


def _start_or_step_attempt(
    *,
    sample_index: Any,
    episode_state: _EpisodeState,
    task: Task[Any],
    agent: Agent,
    callback_handler: CallbackHandler,
    callback_dict: Mapping[str, Any],
    session_list: Sequence[Session],
    current_session: Optional[Session],
    current_callback_args: Optional[CallbackArguments],
) -> tuple[Session, CallbackArguments]:
    if current_session is None:
        _set_task_replay_controls(
            task, episode_state.replay_mode, episode_state.replay_sample_count
        )
        apply_os_rl_budget_profile(agent, task, episode_state.budget_name)
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
        if episode_state.replay_mode != "none":
            episode_state.retrieval_injections += 1
    assert current_callback_args is not None
    apply_os_rl_budget_profile(agent, task, episode_state.budget_name)
    if current_session.sample_status == SampleStatus.RUNNING:
        if current_callback_args.session_controller.should_agent_inference:
            agent.inference(current_session)
            callback_handler.on_agent_inference(current_callback_args)
        if current_callback_args.session_controller.should_task_interact:
            task.interact(current_session)
            callback_handler.on_task_interact(current_callback_args)
    return current_session, current_callback_args


def _finalize_episode_session(
    *,
    task: Task[Any],
    callback_handler: CallbackHandler,
    session: Session,
    callback_args: CallbackArguments,
    force_incorrect_without_evaluation: bool,
) -> Session:
    if force_incorrect_without_evaluation:
        _abort_current_task_attempt(task)
        if session.sample_status == SampleStatus.RUNNING:
            session.sample_status = SampleStatus.TASK_LIMIT_REACHED
        if session.finish_reason is None:
            session.finish_reason = (
                "RL controller terminated without accepting a completed attempt."
            )
        session.evaluation_record.outcome = SessionEvaluationOutcome.INCORRECT
        return session
    if callback_args.session_controller.should_task_complete:
        task.complete(session)
        callback_handler.on_task_complete(callback_args)
    return session


def run_os_inference_rl_episode(
    *,
    sample_index: Any,
    task: Task[Any],
    agent: Agent,
    callback_handler: CallbackHandler,
    callback_dict: Mapping[str, Any],
    session_list: Sequence[Session],
    cost_tracker: Any,
    controller: MaskedDQNInferenceController,
    output_dir: str,
    runtime_state: Mapping[str, Any],
    max_decisions_per_sample: int,
    max_attempts: int,
    max_retrieval_injections: int,
    max_verifications: int,
    budget_target_usd: float,
    cost_lambda: float,
    verifier_mode: str,
) -> RLEpisodeResult:
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    log_path = os.path.join(metrics_dir, "inference_rl.jsonl")
    state_path = os.path.join(metrics_dir, "inference_rl_state.pt")
    summary_path = os.path.join(metrics_dir, "inference_rl_summary.json")

    episode_state = _EpisodeState()
    current_session: Optional[Session] = None
    current_callback_args: Optional[CallbackArguments] = None
    dataset_item = task.get_dataset_item_for_sample(sample_index)
    start_call_idx = len(_get_cost_tracker_calls(cost_tracker))
    transitions: list[RLEpisodeTransition] = []
    decision_records: list[dict[str, Any]] = []
    action_counts = {name: 0 for name in OS_RL_ACTIONS}
    final_session: Optional[Session] = None
    final_callback_args: Optional[CallbackArguments] = None

    for decision_step in range(max(1, int(max_decisions_per_sample))):
        episode_state.decision_step = decision_step
        calls_now = len(_get_cost_tracker_calls(cost_tracker))
        episode_state.cumulative_cost_usd = _get_call_window_cost_usd(
            cost_tracker, start_call_idx, calls_now
        )
        if current_session is not None:
            episode_state.last_status = current_session.sample_status.value
            if current_session.sample_status == SampleStatus.RUNNING:
                episode_state.stage = "running"
            else:
                episode_state.stage = "terminal"
        retrieval_summary = _get_retrieval_feature_summary(
            callback_dict=callback_dict,
            task=task,
            current_sample_index=sample_index,
            replay_sample_count=episode_state.replay_sample_count or 1,
            budget_target_usd=budget_target_usd,
        )
        state, feature_values = build_os_rl_state(
            dataset_item=dataset_item,
            episode_state=episode_state,
            current_round=int(getattr(task, "current_round", 0) or 0),
            max_round=int(getattr(task, "max_round", 1) or 1),
            max_decisions=max_decisions_per_sample,
            max_attempts=max_attempts,
            max_retrieval_injections=max_retrieval_injections,
            max_verifications=max_verifications,
            retrieval_summary=retrieval_summary,
            runtime_state=runtime_state,
            budget_target_usd=budget_target_usd,
        )
        valid_mask = valid_os_rl_action_mask(
            episode_state,
            current_session=current_session,
            max_attempts=max_attempts,
            max_retrieval_injections=max_retrieval_injections,
            max_verifications=max_verifications,
        )
        action_idx, action_meta = controller.select_action(state, valid_mask)
        action_name = controller.action_names[action_idx]
        action_counts[action_name] += 1
        done = False
        action_result: dict[str, Any] = {}

        if action_name == "retrieve_none":
            episode_state.replay_mode = "none"
            episode_state.replay_sample_count = None
            episode_state.setup_retrieval_selected = True
        elif action_name == "retrieve_top1":
            if episode_state.stage == "setup":
                episode_state.replay_mode = "retrieved"
                episode_state.replay_sample_count = 1
                episode_state.setup_retrieval_selected = True
                episode_state.force_reason_next = True
            elif current_session is not None:
                action_result = append_memory_hint_to_pending_user(
                    current_session,
                    callback_dict=callback_dict,
                    task=task,
                    agent=agent,
                    sample_index=sample_index,
                    replay_sample_count=1,
                )
                if action_result.get("injected"):
                    episode_state.replay_mode = "retrieved"
                    episode_state.replay_sample_count = 1
                    episode_state.retrieval_injections += 1
                episode_state.force_reason_next = True
        elif action_name == "retrieve_top4":
            if episode_state.stage == "setup":
                episode_state.replay_mode = "retrieved"
                episode_state.replay_sample_count = 4
                episode_state.setup_retrieval_selected = True
                episode_state.force_reason_next = True
            elif current_session is not None:
                action_result = append_memory_hint_to_pending_user(
                    current_session,
                    callback_dict=callback_dict,
                    task=task,
                    agent=agent,
                    sample_index=sample_index,
                    replay_sample_count=4,
                )
                if action_result.get("injected"):
                    episode_state.replay_mode = "retrieved"
                    episode_state.replay_sample_count = 4
                    episode_state.retrieval_injections += 1
                episode_state.force_reason_next = True
        elif action_name.startswith("set_budget_"):
            episode_state.budget_name = action_name.replace("set_budget_", "")
            if episode_state.stage == "setup":
                episode_state.setup_budget_selected = True
                episode_state.force_reason_next = True
            elif episode_state.stage == "running":
                episode_state.force_reason_next = True
            apply_os_rl_budget_profile(agent, task, episode_state.budget_name)
        elif action_name == "reason_step":
            episode_state.force_reason_next = False
            episode_state.reason_steps_taken += 1
            current_session, current_callback_args = _start_or_step_attempt(
                sample_index=sample_index,
                episode_state=episode_state,
                task=task,
                agent=agent,
                callback_handler=callback_handler,
                callback_dict=callback_dict,
                session_list=session_list,
                current_session=current_session,
                current_callback_args=current_callback_args,
            )
            if current_session.sample_status == SampleStatus.RUNNING:
                episode_state.stage = "running"
            else:
                episode_state.stage = "terminal"
            episode_state.last_status = current_session.sample_status.value
        elif action_name == "verify":
            score, uncertainty, verifier_metadata = run_self_critique_verifier(
                agent=agent,
                session=current_session,
                verifier_mode=verifier_mode,
            )
            episode_state.verifier_score = score
            episode_state.verifier_uncertainty = uncertainty
            episode_state.verification_count += 1
            action_result = {"verifier": verifier_metadata}
        elif action_name == "retry":
            if current_session is not None:
                _abort_current_task_attempt(task)
            episode_state.retry_count += 1
            episode_state.attempt_index += 1
            episode_state.stage = "setup"
            episode_state.last_status = "initial"
            episode_state.replay_mode = "none"
            episode_state.replay_sample_count = None
            episode_state.budget_name = "medium"
            episode_state.setup_retrieval_selected = False
            episode_state.setup_budget_selected = False
            episode_state.force_reason_next = False
            episode_state.reason_steps_taken = 0
            current_session = None
            current_callback_args = None
        elif action_name == "terminate":
            if current_session is None:
                current_session = Session(
                    task_name=task.task_name, sample_index=sample_index
                )
                current_session.sample_status = SampleStatus.TASK_LIMIT_REACHED
                current_session.finish_reason = (
                    "RL controller terminated before starting an attempt."
                )
                current_session.evaluation_record.outcome = (
                    SessionEvaluationOutcome.INCORRECT
                )
                current_callback_args = CallbackArguments(
                    current_session=current_session,
                    task=task,
                    agent=agent,
                    session_list=session_list,
                )
                episode_state.forced_terminate = True
            else:
                force_incorrect = (
                    current_session.sample_status != SampleStatus.COMPLETED
                )
                episode_state.forced_terminate = bool(force_incorrect)
                assert current_callback_args is not None
                current_session = _finalize_episode_session(
                    task=task,
                    callback_handler=callback_handler,
                    session=current_session,
                    callback_args=current_callback_args,
                    force_incorrect_without_evaluation=force_incorrect,
                )
            final_session = current_session
            final_callback_args = current_callback_args
            done = True

        calls_after = len(_get_cost_tracker_calls(cost_tracker))
        episode_state.cumulative_cost_usd = _get_call_window_cost_usd(
            cost_tracker, start_call_idx, calls_after
        )
        if current_session is not None:
            episode_state.last_status = current_session.sample_status.value
            episode_state.stage = (
                "running"
                if current_session.sample_status == SampleStatus.RUNNING
                else "terminal"
            )
        next_retrieval_summary = _get_retrieval_feature_summary(
            callback_dict=callback_dict,
            task=task,
            current_sample_index=sample_index,
            replay_sample_count=episode_state.replay_sample_count or 1,
            budget_target_usd=budget_target_usd,
        )
        next_state, _ = build_os_rl_state(
            dataset_item=dataset_item,
            episode_state=episode_state,
            current_round=int(getattr(task, "current_round", 0) or 0),
            max_round=int(getattr(task, "max_round", 1) or 1),
            max_decisions=max_decisions_per_sample,
            max_attempts=max_attempts,
            max_retrieval_injections=max_retrieval_injections,
            max_verifications=max_verifications,
            retrieval_summary=next_retrieval_summary,
            runtime_state=runtime_state,
            budget_target_usd=budget_target_usd,
        )
        next_valid_mask = valid_os_rl_action_mask(
            episode_state,
            current_session=current_session,
            max_attempts=max_attempts,
            max_retrieval_injections=max_retrieval_injections,
            max_verifications=max_verifications,
        )
        transitions.append(
            RLEpisodeTransition(
                state=state,
                action_idx=action_idx,
                next_state=next_state,
                next_valid_action_mask=next_valid_mask,
                done=done,
            )
        )
        decision_records.append(
            {
                "sample_index": sample_index,
                "sample_index_raw": str(sample_index),
                "decision_step": decision_step,
                "attempt_index": int(episode_state.attempt_index),
                "stage_before": feature_values,
                "state_feature_names": list(OS_RL_FEATURE_NAMES),
                "state_features": feature_values,
                "valid_action_mask": {
                    name: bool(valid_mask[idx])
                    for idx, name in enumerate(controller.action_names)
                },
                "selected_action": action_name,
                "selected_action_idx": int(action_idx),
                "action_meta": action_meta,
                "action_result": action_result,
                "replay_mode": episode_state.replay_mode,
                "replay_sample_count": episode_state.replay_sample_count,
                "budget_name": episode_state.budget_name,
                "verifier_score": float(episode_state.verifier_score),
                "verifier_uncertainty": float(episode_state.verifier_uncertainty),
                "cumulative_cost_usd": float(episode_state.cumulative_cost_usd),
            }
        )
        if done:
            break

    if final_session is None:
        if current_session is not None and current_callback_args is not None:
            force_incorrect = current_session.sample_status != SampleStatus.COMPLETED
            episode_state.forced_terminate = bool(force_incorrect)
            final_session = _finalize_episode_session(
                task=task,
                callback_handler=callback_handler,
                session=current_session,
                callback_args=current_callback_args,
                force_incorrect_without_evaluation=force_incorrect,
            )
            final_callback_args = current_callback_args
        else:
            final_session = Session(task_name=task.task_name, sample_index=sample_index)
            final_session.sample_status = SampleStatus.TASK_LIMIT_REACHED
            final_session.finish_reason = (
                "RL controller reached decision cap before starting an attempt."
            )
            final_session.evaluation_record.outcome = SessionEvaluationOutcome.INCORRECT
            final_callback_args = CallbackArguments(
                current_session=final_session,
                task=task,
                agent=agent,
                session_list=session_list,
            )
            episode_state.forced_terminate = True

    assert final_callback_args is not None
    end_call_idx = len(_get_cost_tracker_calls(cost_tracker))
    total_cost_usd = _get_call_window_cost_usd(
        cost_tracker, start_call_idx, end_call_idx
    )
    correct = (
        final_session.evaluation_record.outcome == SessionEvaluationOutcome.CORRECT
    )
    validation_failed = (
        final_session.sample_status == SampleStatus.AGENT_VALIDATION_FAILED
    )
    final_reward = compute_final_reward(
        correct=bool(correct),
        total_cost_usd=total_cost_usd,
        budget_target_usd=budget_target_usd,
        cost_lambda=cost_lambda,
        validation_failed=bool(validation_failed),
        retry_count=episode_state.retry_count,
        forced_terminate=episode_state.forced_terminate,
    )
    update_meta = controller.observe_episode(transitions, final_reward)
    controller.save_state(state_path)

    summary = {
        "sample_index": sample_index,
        "correct": bool(correct),
        "sample_status": final_session.sample_status.value,
        "evaluation_outcome": final_session.evaluation_record.outcome.value,
        "total_cost_usd": float(total_cost_usd),
        "total_cost_normalized": float(total_cost_usd)
        / max(float(budget_target_usd), 1e-9),
        "reward": float(final_reward),
        "decision_count": len(decision_records),
        "retry_count": int(episode_state.retry_count),
        "forced_terminate": bool(episode_state.forced_terminate),
        "action_counts": action_counts,
        "epsilon_after": float(controller.epsilon),
        "training": update_meta,
    }
    with open(log_path, "a") as f:
        for record in decision_records:
            record.update(
                {
                    "final_correct": bool(correct),
                    "final_sample_status": final_session.sample_status.value,
                    "final_evaluation_outcome": final_session.evaluation_record.outcome.value,
                    "sample_cost_usd": float(total_cost_usd),
                    "sample_cost_normalized": float(summary["total_cost_normalized"]),
                    "reward": float(final_reward),
                }
            )
            f.write(json.dumps(record) + "\n")
    write_inference_rl_summary(
        log_path=log_path,
        summary_path=summary_path,
        controller=controller,
    )
    return RLEpisodeResult(
        session=final_session,
        callback_args=final_callback_args,
        summary=summary,
    )


def write_inference_rl_summary(
    *,
    log_path: str,
    summary_path: str,
    controller: Optional[MaskedDQNInferenceController] = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    by_sample: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_sample[str(row.get("sample_index_raw", row.get("sample_index")))] = row
    sample_rows = list(by_sample.values())
    total = len(sample_rows)
    accuracy = (
        sum(1 for row in sample_rows if bool(row.get("final_correct", False))) / total
        if total > 0
        else 0.0
    )
    mean_cost = (
        sum(float(row.get("sample_cost_usd", 0.0) or 0.0) for row in sample_rows)
        / total
        if total > 0
        else 0.0
    )
    action_counts = {name: 0 for name in OS_RL_ACTIONS}
    for row in rows:
        selected_action = str(row.get("selected_action", ""))
        if selected_action in action_counts:
            action_counts[selected_action] += 1
    summary = {
        "enabled": True,
        "samples": total,
        "decision_rows": len(rows),
        "accuracy": accuracy,
        "mean_cost_usd": mean_cost,
        "action_counts": action_counts,
        "epsilon": None if controller is None else float(controller.epsilon),
        "episodes_seen": None if controller is None else int(controller.episodes_seen),
        "training_steps": (
            None if controller is None else int(controller.training_steps)
        ),
        "algorithm": "masked_double_dqn",
        "feature_names": list(OS_RL_FEATURE_NAMES),
        "action_names": list(OS_RL_ACTIONS),
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary
