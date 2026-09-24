from __future__ import annotations

import logging
import re
from statistics import mean
from typing import Any, Optional

from src.callbacks.callback import CallbackArguments
from src.typings import Role, Session

from .previous_sample_utilization_callback import PreviousSampleUtilizationCallback

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "which",
    "with",
}


class RetrievalPreviousSampleUtilizationCallback(PreviousSampleUtilizationCallback):
    """
    Replay callback that retrieves the most relevant past successful trajectories
    instead of always injecting the most recent ones.

    Design:
    - keep a larger memory bank of past successful sessions
    - at prompt-construction time, score each memory item against the current
      DBBench sample
    - inject the top-k retrieved trajectories

    In replay-gated mode, supported values are:
    - "none":      no replay
    - "full":      inject the most recent `full_replay_sample_count` sessions
    - "retrieved": inject the top `retrieved_sample_count` retrieved sessions
    """

    def __init__(
        self,
        original_first_user_prompt: str,
        utilized_sample_count: int,
        enable_success_cards: bool = False,
        replay_render_mode: str = "raw",
        trimmed_result_row_limit: int = 5,
        trimmed_result_char_limit: int = 240,
        retrieved_sample_count: int = 16,
        full_replay_sample_count: int = 16,
        lexical_weight: float = 0.35,
        skill_weight: float = 0.50,
        table_weight: float = 0.15,
        anchor_weight: float = 0.0,
    ) -> None:
        super().__init__(
            original_first_user_prompt=original_first_user_prompt,
            utilized_sample_count=utilized_sample_count,
            enable_success_cards=enable_success_cards,
            replay_render_mode=replay_render_mode,
            trimmed_result_row_limit=trimmed_result_row_limit,
            trimmed_result_char_limit=trimmed_result_char_limit,
        )
        assert retrieved_sample_count > 0
        assert full_replay_sample_count > 0
        self.retrieved_sample_count = int(retrieved_sample_count)
        self.full_replay_sample_count = int(full_replay_sample_count)
        self.lexical_weight = float(lexical_weight)
        self.skill_weight = float(skill_weight)
        self.table_weight = float(table_weight)
        self.anchor_weight = float(anchor_weight)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        token_list = re.findall(r"[a-z0-9_]+", text.lower())
        return {
            token
            for token in token_list
            if len(token) > 1 and token not in _STOPWORDS
        }

    @staticmethod
    def _extract_dataset_item(task: Any, sample_index: Any) -> Any:
        getter = getattr(task, "get_dataset_item_for_sample", None)
        if callable(getter):
            return getter(sample_index)
        dataset = getattr(task, "dataset", None)
        if isinstance(dataset, dict):
            return dataset[sample_index]
        raise AttributeError(
            "Task does not expose get_dataset_item_for_sample; retrieval replay "
            "needs read-only dataset access before reset()."
        )

    @staticmethod
    def _jaccard(set1: set[str], set2: set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        return float(len(set1 & set2)) / float(max(len(set1 | set2), 1))

    @staticmethod
    def _extract_os_anchor_tokens(dataset_item: Any) -> set[str]:
        """
        OS Interaction-specific retrieval anchor.

        We avoid any evaluation-time leakage and instead capture filesystem-ish
        entities and setup context that often define task families better than
        generic lexical overlap alone: paths, filenames/extensions, usernames,
        group names, and initialization-script artifacts.
        """
        instruction = str(getattr(dataset_item, "instruction", "") or "")
        init_item = getattr(dataset_item, "initialization_command_item", None)
        init_script = str(getattr(init_item, "script", "") or "")
        combined = f"{instruction}\n{init_script}"
        anchor_tokens: set[str] = set()
        for token in re.findall(r"/[A-Za-z0-9._/-]+", combined):
            anchor_tokens.add(token.lower())
            for piece in re.findall(r"[a-z0-9._-]+", token.lower()):
                if len(piece) > 1 and piece not in _STOPWORDS:
                    anchor_tokens.add(piece)
        for token in re.findall(r"[A-Za-z0-9_-]+\.[A-Za-z0-9._-]+", combined):
            lowered = token.lower()
            anchor_tokens.add(lowered)
            for piece in re.findall(r"[a-z0-9._-]+", lowered):
                if len(piece) > 1 and piece not in _STOPWORDS:
                    anchor_tokens.add(piece)
        return anchor_tokens

    def _score_session(
        self,
        *,
        current_item: Any,
        memory_item: Any,
        session_index: int,
        total_session_count: int,
    ) -> float:
        current_instruction = str(getattr(current_item, "instruction", ""))
        memory_instruction = str(getattr(memory_item, "instruction", ""))
        current_tokens = self._tokenize(current_instruction)
        memory_tokens = self._tokenize(memory_instruction)
        lexical_score = self._jaccard(current_tokens, memory_tokens)

        current_skills = set(getattr(current_item, "skill_list", []) or [])
        memory_skills = set(getattr(memory_item, "skill_list", []) or [])
        skill_score = self._jaccard(current_skills, memory_skills)

        current_table = str(getattr(getattr(current_item, "table_info", None), "name", ""))
        memory_table = str(getattr(getattr(memory_item, "table_info", None), "name", ""))
        table_score = 1.0 if current_table and current_table == memory_table else 0.0

        current_anchor_tokens = self._extract_os_anchor_tokens(current_item)
        memory_anchor_tokens = self._extract_os_anchor_tokens(memory_item)
        anchor_score = self._jaccard(current_anchor_tokens, memory_anchor_tokens)

        # Small recency bonus for tie-breaking while staying mostly relevance-driven.
        recency_bonus = 0.02 * float(session_index + 1) / float(max(total_session_count, 1))

        return (
            self.lexical_weight * lexical_score
            + self.skill_weight * skill_score
            + self.table_weight * table_score
            + self.anchor_weight * anchor_score
            + recency_bonus
        )

    def _get_scored_session_list(
        self,
        *,
        task: Any,
        current_sample_index: Any,
        session_list: Optional[list[Session]] = None,
    ) -> list[tuple[float, int, Session]]:
        current_item = self._extract_dataset_item(task, current_sample_index)
        scored_session_list: list[tuple[float, int, Session]] = []
        candidate_session_list = (
            session_list if session_list is not None else self.utilized_session_list
        )
        total_session_count = len(candidate_session_list)
        for session_index, session in enumerate(candidate_session_list):
            memory_item = self._extract_dataset_item(task, session.sample_index)
            score = self._score_session(
                current_item=current_item,
                memory_item=memory_item,
                session_index=session_index,
                total_session_count=total_session_count,
            )
            scored_session_list.append((score, session_index, session))
        scored_session_list.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored_session_list

    def _select_retrieved_sessions(
        self,
        *,
        task: Any,
        current_sample_index: Any,
        replay_sample_count: int | None = None,
    ) -> list[Session]:
        selected = self._select_retrieved_session_tuples(
            task=task,
            current_sample_index=current_sample_index,
            replay_sample_count=replay_sample_count,
        )
        logger.info(
            "Retrieval replay selected sample_indices=%s",
            [session.sample_index for _, _, session in selected],
        )
        return [session for _, _, session in selected]

    def _select_retrieved_session_tuples(
        self,
        *,
        task: Any,
        current_sample_index: Any,
        replay_sample_count: int | None = None,
        session_list: Optional[list[Session]] = None,
    ) -> list[tuple[float, int, Session]]:
        target_count = (
            int(replay_sample_count)
            if replay_sample_count is not None
            else self.retrieved_sample_count
        )
        target_count = max(1, target_count)
        candidate_session_list = (
            session_list if session_list is not None else self.utilized_session_list
        )
        if len(candidate_session_list) == 0:
            return []

        scored_session_list = self._get_scored_session_list(
            task=task,
            current_sample_index=current_sample_index,
            session_list=candidate_session_list,
        )
        return scored_session_list[:target_count]

    def get_retrieval_feature_summary(
        self,
        *,
        task: Any,
        current_sample_index: Any,
        replay_sample_count: Optional[int] = None,
    ) -> dict[str, Any]:
        selected = self._select_retrieved_session_tuples(
            task=task,
            current_sample_index=current_sample_index,
            replay_sample_count=replay_sample_count,
        )
        selected_scores = [float(score) for score, _, _ in selected]
        rounds_list: list[int] = []
        cost_list: list[float] = []
        for _, _, session in selected:
            rounds_used, _, runtime_cost = self._get_session_runtime_metadata(session)
            if isinstance(rounds_used, int):
                rounds_list.append(int(rounds_used))
            if isinstance(runtime_cost, (int, float)):
                cost_list.append(float(runtime_cost))
        summary = {
            "retrieved_example_count": len(selected),
            "top1_similarity_score": selected_scores[0] if selected_scores else 0.0,
            "avg_similarity_score": mean(selected_scores) if selected_scores else 0.0,
            "avg_rounds_in_retrieved": mean(rounds_list) if rounds_list else 0.0,
            "avg_cost_in_retrieved_usd": mean(cost_list) if cost_list else 0.0,
            "retrieved_cost_coverage": (
                float(len(cost_list)) / float(len(selected)) if selected else 0.0
            ),
            "selected_sample_indices": [session.sample_index for _, _, session in selected],
        }
        logger.info(
            "[RetrievalSummary] sample=%s count=%s top1=%.4f avg_sim=%.4f avg_rounds=%.3f avg_cost_usd=%.6f cost_coverage=%.2f selected=%s",
            current_sample_index,
            summary["retrieved_example_count"],
            summary["top1_similarity_score"],
            summary["avg_similarity_score"],
            summary["avg_rounds_in_retrieved"],
            summary["avg_cost_in_retrieved_usd"],
            summary["retrieved_cost_coverage"],
            summary["selected_sample_indices"],
        )
        return summary

    def _render_session_block(self, session: Session, agent_role_dict: dict[Any, str]) -> str:
        return super()._render_session_block(session, agent_role_dict)

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        assert callback_args.current_session.chat_history.get_value_length() == 0

        task = callback_args.session_context.task
        replay_mode: str = getattr(task, "_copal_replay_mode", "retrieved")
        replay_sample_count_override = getattr(
            task, "_copal_replay_sample_count", None
        )

        if replay_mode == "none":
            first_user_prompt = self.original_first_user_prompt.replace(self.pattern, "")
            task.chat_history_item_factory.set(0, Role.USER, first_user_prompt)
            return

        agent_role_dict = callback_args.session_context.agent.get_role_dict()
        selected_session_list: list[Session]
        if replay_mode == "full":
            full_replay_sample_count = (
                int(replay_sample_count_override)
                if replay_sample_count_override is not None
                else self.full_replay_sample_count
            )
            full_replay_sample_count = max(1, full_replay_sample_count)
            selected_session_list = self.utilized_session_list[
                -full_replay_sample_count:
            ]
        else:
            selected_session_list = self._select_retrieved_sessions(
                task=task,
                current_sample_index=callback_args.current_session.sample_index,
                replay_sample_count=replay_sample_count_override,
            )

        example_text = "\n"
        for session in selected_session_list:
            example_text += self._render_session_block(session, agent_role_dict)

        first_user_prompt = self.original_first_user_prompt.replace(
            self.pattern, example_text
        )
        task.chat_history_item_factory.set(0, Role.USER, first_user_prompt)
