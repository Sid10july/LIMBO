from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from src.tasks.task import DatasetItem
from src.tasks.instance.db_bench.task import DBBenchDatasetItem, DBBenchSkillUtility
from src.typings import ChatHistory, Role

GENERALIZABLE_FEATURE_NAMES: tuple[str, ...] = (
    "bias",
    "prompt_tokens",
    "last_user_tokens",
    "system_prompt_tokens",
    "prompt_scaffold_tokens",
    "last_user_share_of_prompt",
    "skill_count",
    "running_cost_ema_normalized",
    "running_accuracy",
    "previous_sample_cost_normalized",
    "previous_sample_correct",
    "previous_sample_had_error",
    "previous_action_uncertainty",
    "recent_task_limit_rate",
    "cost_regime_delta",
    "replay_advantage_ema",
    "novelty_to_successful_history",
)

RETRIEVAL_CONDITIONED_FEATURE_NAMES: tuple[str, ...] = (
    "retrieved_example_count",
    "top1_similarity_score",
    "avg_similarity_score",
    "avg_rounds_in_retrieved",
    "avg_cost_in_retrieved",
)

DBBENCH_RICHER_FEATURE_NAMES: tuple[str, ...] = (
    "rows",
    "cols",
    "cells",
    "skill_count",
    "instruction_x_skill_count",
    "rows_x_skill_count",
    "cols_x_skill_count",
    "row_bucket_50",
    "row_bucket_200",
    "instruction_bucket_300",
    "join_kw",
    "group_kw",
    "having_kw",
    "order_kw",
    "distinct_kw",
    "multi_col_mentions",
    "table_name_mentioned",
    "column_mention_ratio",
    "join_long_interaction",
)


@dataclass(frozen=True)
class BudgetPrimitive:
    name: str
    token_budget: int
    max_round: int
    tool_budget: int
    stop_enabled: bool
    replay_mode: str = "full"
    replay_sample_count: int | None = None
    """
    Controls how the replay callback behaves for this action.
    Values:
      "full"       – inject full past trajectories (default, existing behaviour)
      "none"       – skip replay injection entirely (saves all replay tokens)
      "retrieved"  – inject top-k retrieved trajectories from the replay bank
      "compressed" – compress trajectories before injection (requires
                     LLMLinguaPreviousSampleUtilizationCallback in the config)
    """

    @property
    def static_cost_proxy(self) -> float:
        # Benchmark-agnostic proxy used for "cheapest-safe" ranking.
        # No-replay actions get a multiplier discount because they save the
        # entire replay-injection token budget on top of the compute budget.
        base = float(self.token_budget * self.max_round)
        if self.replay_mode == "none":
            return base * 0.5
        if self.replay_mode == "retrieved":
            return base * 0.9
        if self.replay_mode == "compressed":
            return base * 0.75
        return base


# General budget primitives (not benchmark-specific names).
GENERAL_BUDGET_PRIMITIVES: tuple[BudgetPrimitive, ...] = (
    BudgetPrimitive(
        name="budget_64_r1",
        token_budget=64,
        max_round=1,
        tool_budget=0,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_128_r1",
        token_budget=128,
        max_round=1,
        tool_budget=1,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_256_r2",
        token_budget=256,
        max_round=2,
        tool_budget=2,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_512_r3",
        token_budget=512,
        max_round=3,
        tool_budget=4,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_1024_r4",
        token_budget=1024,
        max_round=4,
        tool_budget=4,
        stop_enabled=True,
    ),
)

# Alternative action set that decouples cheaper token budgets from very tight tool caps.
TOOL_FRIENDLY_BUDGET_PRIMITIVES: tuple[BudgetPrimitive, ...] = (
    BudgetPrimitive(
        name="budget_128_r1_t2",
        token_budget=128,
        max_round=1,
        tool_budget=2,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_256_r2_t4",
        token_budget=256,
        max_round=2,
        tool_budget=4,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_384_r2_t4",
        token_budget=384,
        max_round=2,
        tool_budget=4,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
    ),
    BudgetPrimitive(
        name="budget_768_r3_t6",
        token_budget=768,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
    ),
)


# Replay-gated action space.
# Each action bundles a replay decision (none / full / compressed) with a
# compute budget.  The bandit learns both WHEN to replay and HOW MUCH compute
# to spend, rather than always injecting replay unconditionally.
#
# Naming convention:  <replay_mode>_<token>_r<rounds>_t<tools>
#   nr  = no replay
#   fr  = full replay
#   cr  = compressed replay (requires LLMLingua callback in config)
REPLAY_GATED_BUDGET_PRIMITIVES: tuple[BudgetPrimitive, ...] = (
    # --- No-replay arms (cheapest: save all injection tokens) ---
    BudgetPrimitive(
        name="nr_256_r2_t4",
        token_budget=256,
        max_round=2,
        tool_budget=4,
        stop_enabled=True,
        replay_mode="none",
    ),
    BudgetPrimitive(
        name="nr_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="none",
    ),
    # --- Full-replay arms (same as TOOL_FRIENDLY_BUDGET_PRIMITIVES) ---
    BudgetPrimitive(
        name="fr_256_r2_t4",
        token_budget=256,
        max_round=2,
        tool_budget=4,
        stop_enabled=True,
        replay_mode="full",
    ),
    BudgetPrimitive(
        name="fr_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="full",
    ),
    BudgetPrimitive(
        name="fr_768_r3_t6",
        token_budget=768,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="full",
    ),
    # --- Compressed-replay arms (LLMLingua callback required in config) ---
    BudgetPrimitive(
        name="cr_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="compressed",
    ),
    BudgetPrimitive(
        name="cr_768_r3_t6",
        token_budget=768,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="compressed",
    ),
)


# Retrieval-gated action space.
# Uses a retrieval replay callback/config, where:
# - "full" means inject the recent full_replay_sample_count trajectories
# - "retrieved" means inject the top retrieved_sample_count trajectories
# from a larger memory bank.
RETRIEVAL_GATED_BUDGET_PRIMITIVES: tuple[BudgetPrimitive, ...] = (
    BudgetPrimitive(
        name="nr_256_r2_t4",
        token_budget=256,
        max_round=2,
        tool_budget=4,
        stop_enabled=True,
        replay_mode="none",
    ),
    BudgetPrimitive(
        name="nr_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="none",
    ),
    BudgetPrimitive(
        name="fr_256_r2_t4",
        token_budget=256,
        max_round=2,
        tool_budget=4,
        stop_enabled=True,
        replay_mode="full",
    ),
    BudgetPrimitive(
        name="fr_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="full",
    ),
    BudgetPrimitive(
        name="rr_256_r2_t4",
        token_budget=256,
        max_round=2,
        tool_budget=4,
        stop_enabled=True,
        replay_mode="retrieved",
    ),
    BudgetPrimitive(
        name="rr_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="retrieved",
    ),
    BudgetPrimitive(
        name="rr_768_r3_t6",
        token_budget=768,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="retrieved",
    ),
    BudgetPrimitive(
        name="fr_1024_r4_t8",
        token_budget=1024,
        max_round=4,
        tool_budget=8,
        stop_enabled=True,
        replay_mode="full",
    ),
    BudgetPrimitive(
        name="rr_1024_r4_t8",
        token_budget=1024,
        max_round=4,
        tool_budget=8,
        stop_enabled=True,
        replay_mode="retrieved",
    ),
    BudgetPrimitive(
        name="rr_1536_r5_t10",
        token_budget=1536,
        max_round=5,
        tool_budget=10,
        stop_enabled=True,
        replay_mode="retrieved",
    ),
)


# Retrieval-size gated action space.
# Keeps the compute budget fixed across retrieved arms so the controller mainly
# learns how much replay to buy, not whether to mix replay-size with deeper
# inference settings.
RETRIEVAL_SIZE_GATED_BUDGET_PRIMITIVES: tuple[BudgetPrimitive, ...] = (
    BudgetPrimitive(
        name="nr_256_r2_t4",
        token_budget=256,
        max_round=2,
        tool_budget=4,
        stop_enabled=True,
        replay_mode="none",
    ),
    BudgetPrimitive(
        name="nr_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="none",
    ),
    BudgetPrimitive(
        name="rr4_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="retrieved",
        replay_sample_count=4,
    ),
    BudgetPrimitive(
        name="rr8_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="retrieved",
        replay_sample_count=8,
    ),
    BudgetPrimitive(
        name="rr16_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="retrieved",
        replay_sample_count=16,
    ),
)


# Replay-heavy retrieval-size action space.
# Intended for regimes where replay is broadly useful and the controller should
# spend most of its exploration budget on stronger retrieved-replay options.
REPLAY_HEAVY_RETRIEVAL_SIZE_GATED_BUDGET_PRIMITIVES: tuple[BudgetPrimitive, ...] = (
    BudgetPrimitive(
        name="nr_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="none",
    ),
    BudgetPrimitive(
        name="rr4_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="retrieved",
        replay_sample_count=4,
    ),
    BudgetPrimitive(
        name="rr8_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="retrieved",
        replay_sample_count=8,
    ),
    BudgetPrimitive(
        name="rr16_512_r3_t6",
        token_budget=512,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="retrieved",
        replay_sample_count=16,
    ),
    BudgetPrimitive(
        name="rr8_768_r3_t6",
        token_budget=768,
        max_round=3,
        tool_budget=6,
        stop_enabled=True,
        replay_mode="retrieved",
        replay_sample_count=8,
    ),
    BudgetPrimitive(
        name="rr16_1024_r4_t8",
        token_budget=1024,
        max_round=4,
        tool_budget=8,
        stop_enabled=True,
        replay_mode="retrieved",
        replay_sample_count=16,
    ),
)


def _dot(v1: Sequence[float], v2: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))


def _mat_vec(mat: Sequence[Sequence[float]], vec: Sequence[float]) -> list[float]:
    return [_dot(row, vec) for row in mat]


class LinUCB:
    def __init__(self, n_actions: int, feature_dim: int, alpha: float):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.a_inv_list: list[list[list[float]]] = []
        self.b_list: list[list[float]] = []
        for _ in range(n_actions):
            a_inv = [[0.0 for _ in range(feature_dim)] for _ in range(feature_dim)]
            for i in range(feature_dim):
                a_inv[i][i] = 1.0
            self.a_inv_list.append(a_inv)
            self.b_list.append([0.0 for _ in range(feature_dim)])

    def select_action(self, x: list[float]) -> tuple[int, float]:
        best_action = 0
        best_score = float("-inf")
        for action_idx in range(self.n_actions):
            a_inv = self.a_inv_list[action_idx]
            b = self.b_list[action_idx]
            theta = _mat_vec(a_inv, b)
            exploit = _dot(theta, x)
            a_inv_x = _mat_vec(a_inv, x)
            explore = self.alpha * math.sqrt(max(_dot(x, a_inv_x), 0.0))
            score = exploit + explore
            if score > best_score:
                best_action = action_idx
                best_score = score
        return best_action, best_score

    def update(
        self,
        action_idx: int,
        x: list[float],
        reward: float,
        lr_scale: float = 1.0,
    ) -> None:
        a_inv = self.a_inv_list[action_idx]
        b = self.b_list[action_idx]
        a_inv_x = _mat_vec(a_inv, x)
        denom = 1.0 + _dot(x, a_inv_x)
        # Sherman-Morrison update for A_inv where A <- A + x x^T
        for i in range(self.feature_dim):
            for j in range(self.feature_dim):
                a_inv[i][j] -= (a_inv_x[i] * a_inv_x[j]) / denom
        for i in range(self.feature_dim):
            b[i] += (lr_scale * reward) * x[i]


class TwoHeadLinUCB:
    """
    Two-head linear contextual controller.

    Head 1 predicts correctness probability (in [0, 1], clipped).
    Head 2 predicts normalized cost (>= 0, clipped), typically scaled by budget_target.

    Decision rule per action a:
        score(a) = p_correct(a) - lambda * cost(a) + alpha * uncertainty(a)
    """

    def __init__(self, n_actions: int, feature_dim: int, alpha: float):
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.a_inv_list: list[list[list[float]]] = []
        self.b_correct_list: list[list[float]] = []
        self.b_cost_list: list[list[float]] = []
        for _ in range(n_actions):
            a_inv = [[0.0 for _ in range(feature_dim)] for _ in range(feature_dim)]
            for i in range(feature_dim):
                a_inv[i][i] = 1.0
            self.a_inv_list.append(a_inv)
            self.b_correct_list.append([0.0 for _ in range(feature_dim)])
            self.b_cost_list.append([0.0 for _ in range(feature_dim)])

    @staticmethod
    def _sigmoid(value: float) -> float:
        # Stable enough for the current scale of linear outputs.
        if value >= 0:
            exp_neg = math.exp(-value)
            return 1.0 / (1.0 + exp_neg)
        exp_pos = math.exp(value)
        return exp_pos / (1.0 + exp_pos)

    def predict_all(self, x: list[float]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for action_idx in range(self.n_actions):
            a_inv = self.a_inv_list[action_idx]
            theta_correct = _mat_vec(a_inv, self.b_correct_list[action_idx])
            theta_cost = _mat_vec(a_inv, self.b_cost_list[action_idx])
            raw_correct = _dot(theta_correct, x)
            predicted_correct = self._sigmoid(raw_correct)
            predicted_cost = max(_dot(theta_cost, x), 0.0)
            a_inv_x = _mat_vec(a_inv, x)
            uncertainty = math.sqrt(max(_dot(x, a_inv_x), 0.0))
            out.append(
                {
                    "raw_correct": raw_correct,
                    "predicted_correct": predicted_correct,
                    "predicted_cost": predicted_cost,
                    "uncertainty": uncertainty,
                }
            )
        return out

    def select_action(
        self,
        x: list[float],
        lambda_value: float,
        min_predicted_correct: float = 0.0,
        candidate_action_indices: Sequence[int] | None = None,
    ) -> tuple[int, dict[str, float]]:
        predictions = self.predict_all(x)
        if candidate_action_indices is None:
            candidate_action_indices = tuple(range(self.n_actions))
        active_candidates = list(candidate_action_indices)
        if min_predicted_correct > 0.0:
            safe_candidates = [
                action_idx
                for action_idx in candidate_action_indices
                if predictions[action_idx]["predicted_correct"] >= min_predicted_correct
            ]
            if safe_candidates:
                active_candidates = safe_candidates
            else:
                fallback_idx = max(
                    candidate_action_indices,
                    key=lambda idx: (
                        float(predictions[idx]["predicted_correct"]),
                        -float(predictions[idx]["predicted_cost"]),
                    ),
                )
                pred = predictions[fallback_idx]
                return fallback_idx, {
                    "policy": "utility_floor_fallback",
                    "score": float(pred["predicted_correct"]),
                    "utility": float(pred["predicted_correct"]),
                    "raw_correct": pred["raw_correct"],
                    "predicted_correct": pred["predicted_correct"],
                    "predicted_cost": pred["predicted_cost"],
                    "uncertainty": pred["uncertainty"],
                    "min_predicted_correct": float(min_predicted_correct),
                }
        best_action = 0
        best_score = float("-inf")
        best_meta: dict[str, float] = {}
        for action_idx in active_candidates:
            pred = predictions[action_idx]
            predicted_correct = pred["predicted_correct"]
            predicted_cost = pred["predicted_cost"]
            uncertainty = pred["uncertainty"]
            utility = predicted_correct - (lambda_value * predicted_cost)
            score = utility + (self.alpha * uncertainty)
            if score > best_score:
                best_action = action_idx
                best_score = score
                best_meta = {
                    "score": score,
                    "utility": utility,
                    "raw_correct": pred["raw_correct"],
                    "predicted_correct": predicted_correct,
                    "predicted_cost": predicted_cost,
                    "uncertainty": uncertainty,
                    "min_predicted_correct": float(min_predicted_correct),
                }
        return best_action, best_meta

    def select_cheapest_safe_action(
        self,
        x: list[float],
        correctness_threshold: float,
        action_cost_hint: Sequence[float],
        lambda_value: float = 0.0,
        candidate_action_indices: Sequence[int] | None = None,
    ) -> tuple[int, dict[str, float]]:
        """
        Choose an action from the safe set (predicted correctness >= threshold) by
        maximizing utility = p_correct - lambda * cost_hint.
        If none are safe, fallback to the same utility rule over all candidates.
        """
        predictions = self.predict_all(x)
        if candidate_action_indices is None:
            candidate_action_indices = tuple(range(self.n_actions))

        def utility(idx: int) -> float:
            pred = predictions[idx]
            return float(pred["predicted_correct"]) - (
                float(lambda_value) * float(action_cost_hint[idx])
            )

        safe_indices = []
        for idx in candidate_action_indices:
            if predictions[idx]["predicted_correct"] >= correctness_threshold:
                safe_indices.append(idx)
        if safe_indices:
            best_action = max(
                safe_indices,
                key=utility,
            )
            pred = predictions[best_action]
            return best_action, {
                "policy": "cheapest_safe",
                "threshold": correctness_threshold,
                "lambda_value": float(lambda_value),
                "raw_correct": pred["raw_correct"],
                "predicted_correct": pred["predicted_correct"],
                "predicted_cost": pred["predicted_cost"],
                "cost_hint": float(action_cost_hint[best_action]),
                "uncertainty": pred["uncertainty"],
                "utility": utility(best_action),
                "score": utility(best_action),
            }
        fallback_idx = max(candidate_action_indices, key=utility)
        pred = predictions[fallback_idx]
        return fallback_idx, {
            "policy": "fallback_utility",
            "threshold": correctness_threshold,
            "lambda_value": float(lambda_value),
            "raw_correct": pred["raw_correct"],
            "predicted_correct": pred["predicted_correct"],
            "predicted_cost": pred["predicted_cost"],
            "cost_hint": float(action_cost_hint[fallback_idx]),
            "uncertainty": pred["uncertainty"],
            "utility": utility(fallback_idx),
            "score": utility(fallback_idx),
        }

    def select_fixed_tau_action(
        self,
        x: list[float],
        correctness_threshold: float,
        action_cost_hint: Sequence[float],
        candidate_action_indices: Sequence[int] | None = None,
    ) -> tuple[int, dict[str, float]]:
        predictions = self.predict_all(x)
        if candidate_action_indices is None:
            candidate_action_indices = tuple(range(self.n_actions))

        safe_indices = [
            idx
            for idx in candidate_action_indices
            if predictions[idx]["predicted_correct"] >= correctness_threshold
        ]
        if safe_indices:
            best_action = min(
                safe_indices,
                key=lambda idx: (
                    float(action_cost_hint[idx]),
                    -float(predictions[idx]["predicted_correct"]),
                ),
            )
            pred = predictions[best_action]
            return best_action, {
                "policy": "fixed_tau",
                "threshold": correctness_threshold,
                "raw_correct": pred["raw_correct"],
                "predicted_correct": pred["predicted_correct"],
                "predicted_cost": pred["predicted_cost"],
                "cost_hint": float(action_cost_hint[best_action]),
                "uncertainty": pred["uncertainty"],
                "score": float(pred["predicted_correct"]),
            }

        fallback_idx = max(
            candidate_action_indices,
            key=lambda idx: (
                float(predictions[idx]["predicted_correct"]),
                -float(action_cost_hint[idx]),
            ),
        )
        pred = predictions[fallback_idx]
        return fallback_idx, {
            "policy": "fixed_tau_fallback",
            "threshold": correctness_threshold,
            "raw_correct": pred["raw_correct"],
            "predicted_correct": pred["predicted_correct"],
            "predicted_cost": pred["predicted_cost"],
            "cost_hint": float(action_cost_hint[fallback_idx]),
            "uncertainty": pred["uncertainty"],
            "score": float(pred["predicted_correct"]),
        }

    def update(
        self,
        action_idx: int,
        x: list[float],
        correct_label: float,
        cost_label: float,
        correct_lr_scale: float = 1.0,
        cost_lr_scale: float = 1.0,
        use_residual: bool = False,
    ) -> None:
        a_inv = self.a_inv_list[action_idx]
        b_correct = self.b_correct_list[action_idx]
        b_cost = self.b_cost_list[action_idx]
        if use_residual:
            theta_correct = _mat_vec(a_inv, b_correct)
            theta_cost = _mat_vec(a_inv, b_cost)
            current_correct = self._sigmoid(_dot(theta_correct, x))
            current_cost = max(_dot(theta_cost, x), 0.0)
            correct_target = float(correct_label) - float(current_correct)
            cost_target = float(cost_label) - float(current_cost)
        else:
            correct_target = float(correct_label)
            cost_target = float(cost_label)
        a_inv_x = _mat_vec(a_inv, x)
        denom = 1.0 + _dot(x, a_inv_x)
        # Sherman-Morrison update for A_inv where A <- A + x x^T
        for i in range(self.feature_dim):
            for j in range(self.feature_dim):
                a_inv[i][j] -= (a_inv_x[i] * a_inv_x[j]) / denom
        for i in range(self.feature_dim):
            b_correct[i] += (correct_lr_scale * correct_target) * x[i]
            b_cost[i] += (cost_lr_scale * cost_target) * x[i]


class GeneralFeatureAdapter:
    def __init__(self, selected_feature_names: Sequence[str] | None = None):
        self.base_feature_names = GENERALIZABLE_FEATURE_NAMES
        self.available_feature_names = self.base_feature_names + RETRIEVAL_CONDITIONED_FEATURE_NAMES
        if selected_feature_names:
            normalized_feature_names = [name.strip() for name in selected_feature_names]
            invalid_feature_names = [
                name
                for name in normalized_feature_names
                if name not in self.available_feature_names
            ]
            if invalid_feature_names:
                raise ValueError(
                    "Unknown general bandit feature(s): "
                    + ", ".join(sorted(set(invalid_feature_names)))
                )
            self.feature_names = tuple(normalized_feature_names)
        else:
            # Preserve existing default behaviour for general-bandit runs unless
            # a config explicitly opts into retrieval-conditioned features.
            self.feature_names = self.base_feature_names

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @staticmethod
    def _normalize_log_count(value: int | float, *, scale: float = 10.0) -> float:
        return min(math.log1p(max(float(value), 0.0)) / scale, 1.0)

    @staticmethod
    def _normalize_cost_ratio(value: int | float) -> float:
        return min(max(float(value), 0.0) / 2.0, 1.0)

    @staticmethod
    def _chat_history_to_message_list(
        chat_history: ChatHistory,
        role_dict: Mapping[Role, str],
        system_prompt: str,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if len(system_prompt) > 0:
            messages.append({"role": "system", "content": system_prompt})
        for item_index in range(chat_history.get_value_length()):
            chat_history_item = chat_history.get_item_deep_copy(item_index)
            messages.append(
                {
                    "role": role_dict[chat_history_item.role],
                    "content": chat_history_item.content,
                }
            )
        return messages

    @staticmethod
    def _extract_item_text(item: DatasetItem) -> str:
        for field_name in ("instruction", "question"):
            value = getattr(item, field_name, None)
            if value is not None:
                return str(value)
        try:
            return str(item.model_dump())
        except Exception:
            return str(item)

    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9_]+", text.lower())
            if len(token) >= 2
        }

    @staticmethod
    def _max_jaccard_similarity(
        current_tokens: set[str],
        prior_token_sets: Sequence[set[str]],
    ) -> float:
        if not current_tokens or not prior_token_sets:
            return 0.0
        best = 0.0
        for prior_tokens in prior_token_sets:
            if not prior_tokens:
                continue
            denom = len(current_tokens | prior_tokens)
            if denom <= 0:
                continue
            sim = float(len(current_tokens & prior_tokens)) / float(denom)
            if sim > best:
                best = sim
        return min(max(best, 0.0), 1.0)

    def _estimate_prompt_token_count(
        self,
        *,
        item_text: str,
        chat_history: ChatHistory | None,
        system_prompt: str,
        language_model: Any | None,
    ) -> tuple[int, int, int]:
        tokenizer = getattr(language_model, "tokenizer", None)
        role_dict = getattr(language_model, "role_dict", None)
        if (
            tokenizer is not None
            and chat_history is not None
            and role_dict is not None
            and hasattr(tokenizer, "apply_chat_template")
        ):
            try:
                message_list = self._chat_history_to_message_list(
                    chat_history=chat_history,
                    role_dict=role_dict,
                    system_prompt=system_prompt,
                )
                prompt_ids = tokenizer.apply_chat_template(
                    message_list,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(prompt_ids, Sequence):
                    prompt_tokens = len(prompt_ids)
                else:
                    prompt_tokens = int(prompt_ids.shape[-1])
                last_user_content = (
                    chat_history.get_item_deep_copy(-1).content
                    if chat_history.get_value_length() > 0
                    else item_text
                )
                last_user_ids = tokenizer.encode(
                    last_user_content,
                    add_special_tokens=False,
                )
                last_user_tokens = len(last_user_ids)
                system_prompt_tokens = (
                    len(
                        tokenizer.encode(
                            system_prompt,
                            add_special_tokens=False,
                        )
                    )
                    if len(system_prompt) > 0
                    else 0
                )
                return (
                    int(prompt_tokens),
                    int(last_user_tokens),
                    int(system_prompt_tokens),
                )
            except Exception:
                pass
        last_user_content = item_text
        if chat_history is not None and chat_history.get_value_length() > 0:
            try:
                last_user_content = chat_history.get_item_deep_copy(-1).content
            except Exception:
                last_user_content = item_text
        message_lengths = len(system_prompt)
        if chat_history is not None:
            for item_index in range(chat_history.get_value_length()):
                message_lengths += len(chat_history.get_item_deep_copy(item_index).content)
        else:
            message_lengths += len(item_text)
        fallback_prompt_tokens = max(1, message_lengths // 4)
        fallback_last_user_tokens = max(1, len(last_user_content) // 4)
        fallback_system_prompt_tokens = max(0, len(system_prompt) // 4)
        return (
            fallback_prompt_tokens,
            fallback_last_user_tokens,
            fallback_system_prompt_tokens,
        )

    def _build_general_feature_value_map(
        self,
        item: DatasetItem,
        *,
        chat_history: ChatHistory | None = None,
        system_prompt: str = "",
        language_model: Any | None = None,
        runtime_state: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        item_text = self._extract_item_text(item)
        (
            prompt_token_count,
            last_user_token_count,
            system_prompt_token_count,
        ) = self._estimate_prompt_token_count(
            item_text=item_text,
            chat_history=chat_history,
            system_prompt=system_prompt,
            language_model=language_model,
        )
        runtime_state = runtime_state or {}
        running_cost_ema_normalized = self._normalize_cost_ratio(
            float(runtime_state.get("running_cost_ema_normalized", 0.0))
        )
        running_accuracy = min(
            max(float(runtime_state.get("running_accuracy", 0.0)), 0.0),
            1.0,
        )
        previous_sample_cost_normalized = self._normalize_cost_ratio(
            float(runtime_state.get("previous_sample_cost_normalized", 0.0))
        )
        previous_sample_correct = (
            1.0 if bool(runtime_state.get("previous_sample_correct", False)) else 0.0
        )
        previous_sample_had_error = (
            1.0 if bool(runtime_state.get("previous_sample_had_error", False)) else 0.0
        )
        previous_action_uncertainty = min(
            max(float(runtime_state.get("previous_action_uncertainty", 0.0)), 0.0),
            8.0,
        ) / 8.0
        recent_task_limit_rate = min(
            max(float(runtime_state.get("recent_task_limit_rate", 0.0)), 0.0),
            1.0,
        )
        cost_regime_delta = min(
            max(float(runtime_state.get("cost_regime_delta", 0.0)), -1.0),
            1.0,
        )
        replay_advantage_ema = min(
            max(float(runtime_state.get("replay_advantage_ema", 0.0)), -1.0),
            1.0,
        )
        retrieval_feature_summary = runtime_state.get("retrieval_feature_summary", {})
        if not isinstance(retrieval_feature_summary, Mapping):
            retrieval_feature_summary = {}
        retrieved_example_count = self._normalize_log_count(
            float(retrieval_feature_summary.get("retrieved_example_count", 0.0)),
            scale=2.0,
        )
        top1_similarity_score = min(
            max(float(retrieval_feature_summary.get("top1_similarity_score", 0.0)), 0.0),
            1.0,
        )
        avg_similarity_score = min(
            max(float(retrieval_feature_summary.get("avg_similarity_score", 0.0)), 0.0),
            1.0,
        )
        avg_rounds_in_retrieved = min(
            max(float(retrieval_feature_summary.get("avg_rounds_in_retrieved", 0.0)), 0.0),
            8.0,
        ) / 8.0
        avg_cost_in_retrieved = self._normalize_cost_ratio(
            float(retrieval_feature_summary.get("avg_cost_in_retrieved_normalized", 0.0))
        )
        successful_task_token_sets = runtime_state.get("successful_task_token_sets", [])
        novelty_to_successful_history = 1.0
        if isinstance(successful_task_token_sets, Sequence):
            novelty_to_successful_history = 1.0 - self._max_jaccard_similarity(
                self._tokenize_text(item_text),
                [
                    token_set
                    for token_set in successful_task_token_sets
                    if isinstance(token_set, set)
                ],
            )
        novelty_to_successful_history = min(
            max(novelty_to_successful_history, 0.0),
            1.0,
        )
        norm_prompt_tokens = self._normalize_log_count(prompt_token_count)
        norm_last_user_tokens = self._normalize_log_count(last_user_token_count)
        norm_system_prompt_tokens = self._normalize_log_count(system_prompt_token_count)
        prompt_scaffold_tokens = max(prompt_token_count - last_user_token_count, 0)
        norm_prompt_scaffold_tokens = self._normalize_log_count(prompt_scaffold_tokens)
        skill_list: Sequence[str] = []
        get_skill_list = getattr(item, "get_skill_list", None)
        if callable(get_skill_list):
            try:
                skill_list = list(get_skill_list() or [])
            except Exception:
                skill_list = []
        elif hasattr(item, "skill_list"):
            try:
                skill_list = list(getattr(item, "skill_list") or [])
            except Exception:
                skill_list = []
        norm_skill_count = min(max(float(len(skill_list)), 0.0) / 10.0, 1.0)
        last_user_share = min(
            float(last_user_token_count) / float(max(prompt_token_count, 1)),
            1.0,
        )
        return {
            "bias": 1.0,
            "prompt_tokens": norm_prompt_tokens,
            "last_user_tokens": norm_last_user_tokens,
            "system_prompt_tokens": norm_system_prompt_tokens,
            "prompt_scaffold_tokens": norm_prompt_scaffold_tokens,
            "last_user_share_of_prompt": last_user_share,
            "skill_count": norm_skill_count,
            "running_cost_ema_normalized": running_cost_ema_normalized,
            "running_accuracy": running_accuracy,
            "previous_sample_cost_normalized": previous_sample_cost_normalized,
            "previous_sample_correct": previous_sample_correct,
            "previous_sample_had_error": previous_sample_had_error,
            "previous_action_uncertainty": previous_action_uncertainty,
            "recent_task_limit_rate": recent_task_limit_rate,
            "cost_regime_delta": cost_regime_delta,
            "replay_advantage_ema": replay_advantage_ema,
            "novelty_to_successful_history": novelty_to_successful_history,
            "retrieved_example_count": retrieved_example_count,
            "top1_similarity_score": top1_similarity_score,
            "avg_similarity_score": avg_similarity_score,
            "avg_rounds_in_retrieved": avg_rounds_in_retrieved,
            "avg_cost_in_retrieved": avg_cost_in_retrieved,
        }

    def build(
        self,
        item: DatasetItem,
        *,
        chat_history: ChatHistory | None = None,
        system_prompt: str = "",
        language_model: Any | None = None,
        runtime_state: Mapping[str, Any] | None = None,
    ) -> list[float]:
        feature_value_map = self._build_general_feature_value_map(
            item,
            chat_history=chat_history,
            system_prompt=system_prompt,
            language_model=language_model,
            runtime_state=runtime_state,
        )
        return [feature_value_map[name] for name in self.feature_names]


class DBBenchFeatureAdapter(GeneralFeatureAdapter):
    def __init__(
        self,
        use_richer_features: bool = False,
        selected_feature_names: Sequence[str] | None = None,
    ):
        super().__init__(selected_feature_names=None)
        self.skill_list = DBBenchSkillUtility.get_all_skill_list()
        self.use_richer_features = use_richer_features
        self.base_feature_names = GENERALIZABLE_FEATURE_NAMES + RETRIEVAL_CONDITIONED_FEATURE_NAMES
        self.richer_feature_names = DBBENCH_RICHER_FEATURE_NAMES + tuple(
            f"skill:{skill}" for skill in self.skill_list
        )
        available_feature_names = list(self.base_feature_names)
        if self.use_richer_features:
            available_feature_names.extend(self.richer_feature_names)
        self.available_feature_names = tuple(available_feature_names)
        if selected_feature_names:
            normalized_feature_names = [name.strip() for name in selected_feature_names]
            invalid_feature_names = [
                name
                for name in normalized_feature_names
                if name not in self.available_feature_names
            ]
            if invalid_feature_names:
                raise ValueError(
                    "Unknown DBBench bandit feature(s): "
                    + ", ".join(sorted(set(invalid_feature_names)))
                )
            self.feature_names = tuple(normalized_feature_names)
        else:
            self.feature_names = self.available_feature_names

    @staticmethod
    def _normalize_ratio(value: int | float, *, ceiling: float) -> float:
        if ceiling <= 0:
            return 0.0
        return min(max(float(value), 0.0) / ceiling, 1.0)

    def build(
        self,
        item: DBBenchDatasetItem,
        *,
        chat_history: ChatHistory | None = None,
        system_prompt: str = "",
        language_model: Any | None = None,
        runtime_state: Mapping[str, float | int | bool] | None = None,
    ) -> list[float]:
        instruction_len = len(item.instruction)
        instruction_lower = item.instruction.lower()
        row_count = len(item.table_info.row_list)
        col_count = len(item.table_info.column_info_list)
        sample_skills = set(item.skill_list)
        skill_den = max(len(self.skill_list), 1)
        feature_value_map = self._build_general_feature_value_map(
            item,
            chat_history=chat_history,
            system_prompt=system_prompt,
            language_model=language_model,
            runtime_state=runtime_state,
        )
        (
            prompt_token_count,
            last_user_token_count,
            system_prompt_token_count,
        ) = self._estimate_prompt_token_count(
            item_text=item.instruction,
            chat_history=chat_history,
            system_prompt=system_prompt,
            language_model=language_model,
        )
        table_cell_count = row_count * col_count
        norm_instruction = self._normalize_log_count(instruction_len)
        norm_rows = self._normalize_log_count(row_count)
        norm_cols = self._normalize_log_count(col_count)
        norm_cells = self._normalize_log_count(table_cell_count, scale=12.0)
        norm_skill_count = self._normalize_ratio(len(sample_skills), ceiling=float(skill_den))
        feature_value_map["skill_count"] = norm_skill_count
        if self.use_richer_features:
            column_name_mentions = sum(
                1
                for col in item.table_info.column_info_list
                if col.name.lower() in instruction_lower
            )
            table_name_mentioned = (
                1.0 if item.table_info.name.lower() in instruction_lower else 0.0
            )
            column_mention_ratio = min(
                float(column_name_mentions) / float(max(col_count, 1)),
                1.0,
            )
            join_kw = 1.0 if " join " in f" {instruction_lower} " else 0.0
            group_kw = 1.0 if "group by" in instruction_lower else 0.0
            having_kw = 1.0 if "having" in instruction_lower else 0.0
            order_kw = 1.0 if "order by" in instruction_lower else 0.0
            distinct_kw = 1.0 if "distinct" in instruction_lower else 0.0
            multi_col_mentions = 1.0 if column_name_mentions >= 2 else 0.0
            feature_value_map.update(
                {
                    "rows": norm_rows,
                    "cols": norm_cols,
                    "cells": norm_cells,
                    "skill_count": norm_skill_count,
                    "instruction_x_skill_count": norm_instruction * norm_skill_count,
                    "rows_x_skill_count": norm_rows * norm_skill_count,
                    "cols_x_skill_count": norm_cols * norm_skill_count,
                    "row_bucket_50": 1.0 if row_count > 50 else 0.0,
                    "row_bucket_200": 1.0 if row_count > 200 else 0.0,
                    "instruction_bucket_300": 1.0 if instruction_len > 300 else 0.0,
                    "join_kw": join_kw,
                    "group_kw": group_kw,
                    "having_kw": having_kw,
                    "order_kw": order_kw,
                    "distinct_kw": distinct_kw,
                    "multi_col_mentions": multi_col_mentions,
                    "table_name_mentioned": table_name_mentioned,
                    "column_mention_ratio": column_mention_ratio,
                    "join_long_interaction": norm_instruction
                    * (join_kw + group_kw + having_kw + order_kw),
                }
            )
            for skill in self.skill_list:
                feature_value_map[f"skill:{skill}"] = (
                    1.0 if skill in sample_skills else 0.0
                )
        return [feature_value_map[name] for name in self.feature_names]
