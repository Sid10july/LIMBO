from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.callbacks.callback import CallbackArguments
from src.typings import Role, SampleStatus, Session, SessionEvaluationOutcome

from .retrieval_previous_sample_utilization_callback import (
    RetrievalPreviousSampleUtilizationCallback,
)

logger = logging.getLogger(__name__)


class FailureAwareRetrievalPreviousSampleUtilizationCallback(
    RetrievalPreviousSampleUtilizationCallback
):
    """
    Retrieval replay with lightweight deterministic failure knowledge.

    Successful sessions are still injected as trimmed-trace worked examples.
    Additionally, a small retrieved bank of similar failed sessions is converted
    into short cautionary reminders before the examples. This keeps the prompt
    footprint small and avoids any extra LLM extraction cost.
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
        failed_sample_count: int = 64,
        failed_retrieved_sample_count: int = 2,
        include_failure_knowledge: bool = True,
    ) -> None:
        super().__init__(
            original_first_user_prompt=original_first_user_prompt,
            utilized_sample_count=utilized_sample_count,
            enable_success_cards=enable_success_cards,
            replay_render_mode=replay_render_mode,
            trimmed_result_row_limit=trimmed_result_row_limit,
            trimmed_result_char_limit=trimmed_result_char_limit,
            retrieved_sample_count=retrieved_sample_count,
            full_replay_sample_count=full_replay_sample_count,
            lexical_weight=lexical_weight,
            skill_weight=skill_weight,
            table_weight=table_weight,
            anchor_weight=anchor_weight,
        )
        self.failed_sample_count = max(1, int(failed_sample_count))
        self.failed_retrieved_sample_count = max(1, int(failed_retrieved_sample_count))
        self.include_failure_knowledge = bool(include_failure_knowledge)
        self.failed_session_list: list[Session] = []

    def _get_failed_session_list_state_path(self) -> str:
        return self._get_utilized_session_list_state_path().replace(
            "utilized_session_list.json", "failed_session_list.json"
        )

    def restore_state(self) -> None:
        super().restore_state()
        failed_state_path = self._get_failed_session_list_state_path()
        if failed_state_path and json is not None:
            try:
                with open(failed_state_path, "r") as f:
                    self.failed_session_list = [
                        Session.model_validate(session_info_dict)
                        for session_info_dict in json.load(f)
                    ]
            except FileNotFoundError:
                self.failed_session_list = []

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        super().on_task_complete(callback_args)
        current_session = callback_args.current_session
        if (
            current_session.evaluation_record.outcome
            == SessionEvaluationOutcome.CORRECT
            and current_session.sample_status == SampleStatus.COMPLETED
        ):
            return
        self.failed_session_list.append(current_session)
        if len(self.failed_session_list) > self.failed_sample_count:
            self.failed_session_list.pop(0)

    def _extract_last_command_hint(self, session: Session) -> Optional[str]:
        for item_index in range(session.chat_history.get_value_length() - 1, -1, -1):
            item = session.chat_history.get_item_deep_copy(item_index)
            if item.role != Role.AGENT:
                continue
            content = str(item.content or "").strip()
            if not content:
                continue
            if "```bash" in content:
                try:
                    snippet = content.split("```bash", 1)[1].split("```", 1)[0].strip()
                    first_line = snippet.splitlines()[0].strip() if snippet else ""
                    return first_line[:120] if first_line else None
                except Exception:  # noqa: BLE001
                    return None
            act_line = content.splitlines()[0].strip()
            return act_line[:120] if act_line else None
        return None

    def _build_failure_hint(self, session: Session) -> tuple[str, str]:
        rounds_used, _, _ = self._get_session_runtime_metadata(session)
        status = session.sample_status
        if status == SampleStatus.AGENT_VALIDATION_FAILED:
            return (
                "validation_failed",
                "Similar task previously failed validation; follow the exact response "
                "format: `Act: bash` with one fenced bash block or `Act: finish`."
            )
        if status == SampleStatus.TASK_LIMIT_REACHED:
            return (
                "task_limit",
                f"Similar task previously ran out of rounds after about {rounds_used} "
                "steps; plan a short command sequence up front, prioritize actions "
                "that directly satisfy the requested filesystem constraints, and "
                "avoid redundant exploration or repeated checks unless they verify a "
                "required condition."
            )
        if (
            status == SampleStatus.COMPLETED
            and session.evaluation_record.outcome != SessionEvaluationOutcome.CORRECT
        ):
            return (
                "completed_incorrect",
                "Similar task previously completed but was still incorrect; before "
                "finishing, verify every requested constraint (ownership, permissions, "
                "paths, and file contents)."
            )
        last_command_hint = self._extract_last_command_hint(session)
        if last_command_hint:
            return (
                "command_repeat_risk",
                "A similar past attempt failed; double-check before repeating command: "
                f"`{last_command_hint}`."
            )
        return (
            "generic_failure",
            "A similar past attempt failed; prefer a shorter path to a verifiable final state.",
        )

    def _build_failure_knowledge_text(
        self,
        *,
        task: Any,
        current_sample_index: Any,
    ) -> tuple[str, list[str]]:
        if not self.include_failure_knowledge or len(self.failed_session_list) == 0:
            return "", []
        selected = self._select_retrieved_session_tuples(
            task=task,
            current_sample_index=current_sample_index,
            replay_sample_count=self.failed_retrieved_sample_count,
            session_list=self.failed_session_list,
        )
        if len(selected) == 0:
            return "", []
        lines = ["Failure reminders from similar past tasks:"]
        hint_type_list: list[str] = []
        for _, _, session in selected:
            hint_type, hint_text = self._build_failure_hint(session)
            hint_type_list.append(hint_type)
            lines.append(f"- {hint_text}")
        return "\n".join(lines) + "\n\n", hint_type_list

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        assert callback_args.current_session.chat_history.get_value_length() == 0

        task = callback_args.session_context.task
        replay_mode: str = getattr(task, "_copal_replay_mode", "retrieved")
        replay_sample_count_override = getattr(task, "_copal_replay_sample_count", None)

        if replay_mode == "none":
            first_user_prompt = self.original_first_user_prompt.replace(self.pattern, "")
            task.chat_history_item_factory.set(0, Role.USER, first_user_prompt)
            return

        failure_text, failure_hint_types = self._build_failure_knowledge_text(
            task=task, current_sample_index=callback_args.current_session.sample_index
        )
        if failure_hint_types:
            logger.info(
                "[FailureKnowledge] sample=%s hint_types=%s failed_bank_size=%s",
                callback_args.current_session.sample_index,
                failure_hint_types,
                len(self.failed_session_list),
            )

        agent_role_dict = callback_args.session_context.agent.get_role_dict()
        selected_session_list: list[Session]
        if replay_mode == "full":
            full_replay_sample_count = (
                int(replay_sample_count_override)
                if replay_sample_count_override is not None
                else self.full_replay_sample_count
            )
            full_replay_sample_count = max(1, full_replay_sample_count)
            selected_session_list = self.utilized_session_list[-full_replay_sample_count:]
        else:
            selected_session_list = self._select_retrieved_sessions(
                task=task,
                current_sample_index=callback_args.current_session.sample_index,
                replay_sample_count=replay_sample_count_override,
            )

        example_text = "\n" + failure_text
        for session in selected_session_list:
            example_text += self._render_session_block(session, agent_role_dict)

        first_user_prompt = self.original_first_user_prompt.replace(self.pattern, example_text)
        task.chat_history_item_factory.set(0, Role.USER, first_user_prompt)

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        super().on_state_save(callback_args)
        with open(self._get_failed_session_list_state_path(), "w") as f:
            json.dump([s.model_dump() for s in self.failed_session_list], f, indent=2)
