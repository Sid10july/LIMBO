import os
from typing import Optional, Any, Mapping
import json
import logging
import ast

from src.callbacks.callback import Callback, CallbackArguments
from src.typings import (
    Session,
    Role,
    SessionEvaluationOutcome,
    SampleStatus,
    ChatHistoryItem,
    TaskName,
)
from .success_card import SuccessCard, extract_success_card

logger = logging.getLogger(__name__)


class PreviousSampleUtilizationCallback(Callback):
    def __init__(
        self,
        original_first_user_prompt: str,
        utilized_sample_count: int,
        enable_success_cards: bool = False,
        replay_render_mode: str = "raw",
        trimmed_result_row_limit: int = 5,
        trimmed_result_char_limit: int = 240,
    ):
        super().__init__()
        self.original_first_user_prompt = original_first_user_prompt
        self.pattern = "{previous_sample_utilization_target_position}"
        pattern_count = self.original_first_user_prompt.count(self.pattern)
        if pattern_count == 0:
            insertion = f"\n\n{self.pattern}\n\n"
            needle = "Now, I will give you the question that you need to solve."
            if needle in self.original_first_user_prompt:
                self.original_first_user_prompt = self.original_first_user_prompt.replace(
                    needle,
                    f"{self.pattern}\n\n{needle}",
                    1,
                )
            else:
                self.original_first_user_prompt = (
                    self.original_first_user_prompt.rstrip() + insertion
                )
            logger.warning(
                "Replay prompt marker was missing; inserted %s automatically.",
                self.pattern,
            )
        else:
            assert pattern_count == 1
        assert utilized_sample_count > 0
        self.utilized_sample_count = utilized_sample_count
        self.utilized_session_list: list[Session] = []
        self.enable_success_cards = bool(enable_success_cards)
        assert replay_render_mode in {"raw", "success_card", "trimmed_trace"}
        self.replay_render_mode = str(replay_render_mode)
        self.trimmed_result_row_limit = max(1, int(trimmed_result_row_limit))
        self.trimmed_result_char_limit = max(64, int(trimmed_result_char_limit))
        self.success_card_dict: dict[str, SuccessCard] = {}
        self.success_card_extraction_summary: dict[str, Any] = {
            "count": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
            "by_sample": {},
        }

    def _get_utilized_session_list_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "utilized_session_list.json")

    def _get_success_card_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "success_card_dict.json")

    def _get_success_card_summary_state_path(self) -> str:
        return os.path.join(self.get_state_dir(), "success_card_extraction_summary.json")

    def restore_state(self) -> None:
        self.utilized_session_list = [
            Session.model_validate(session_info_dict)
            for session_info_dict in json.load(
                open(self._get_utilized_session_list_state_path(), "r")
            )
        ]
        success_card_state_path = self._get_success_card_state_path()
        if os.path.exists(success_card_state_path):
            self.success_card_dict = {
                sample_index: SuccessCard.model_validate(card_dict)
                for sample_index, card_dict in json.load(
                    open(success_card_state_path, "r")
                ).items()
            }
        success_card_summary_state_path = self._get_success_card_summary_state_path()
        if os.path.exists(success_card_summary_state_path):
            self.success_card_extraction_summary = json.load(
                open(success_card_summary_state_path, "r")
            )

    @classmethod
    def is_unique(cls) -> bool:
        return True

    def on_task_complete(self, callback_args: CallbackArguments) -> None:
        # Get the session that just completed.
        current_session = callback_args.current_session
        if (
            current_session.evaluation_record.outcome
            == SessionEvaluationOutcome.CORRECT
            and current_session.sample_status == SampleStatus.COMPLETED
        ):
            self.utilized_session_list.append(current_session)
        if len(self.utilized_session_list) > self.utilized_sample_count:
            removed_session = self.utilized_session_list.pop(0)
            self.success_card_dict.pop(str(removed_session.sample_index), None)

    def _get_cost_tracker(self, callback_args: CallbackArguments) -> Any:
        agent = callback_args.session_context.agent
        cost_tracker = getattr(agent, "cost_tracker", None)
        if cost_tracker is None:
            llm_obj = getattr(agent, "_language_model", None)
            if llm_obj is not None:
                cost_tracker = getattr(llm_obj, "cost_tracker", None)
        return cost_tracker

    def _get_session_runtime_metadata(self, session: Session) -> tuple[Optional[int], Optional[int], Optional[float]]:
        detail = session.evaluation_record.detail_dict or {}
        prompt_tokens = detail.get("cost_input_tokens")
        completion_tokens = detail.get("cost_output_tokens")
        total_tokens: Optional[int]
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = int(prompt_tokens + completion_tokens)
        else:
            total_tokens = None
        cost_value = detail.get("cost_usd")
        runtime_cost = float(cost_value) if isinstance(cost_value, (int, float)) else None
        rounds_used = 0
        for item_index in range(3, session.chat_history.get_value_length()):
            item = session.chat_history.get_item_deep_copy(item_index)
            if item.role == Role.AGENT and item.content != "":
                rounds_used += 1
        return rounds_used, total_tokens, runtime_cost

    def _build_dbbench_schema_text(self, dataset_item: Any) -> str:
        table_info = getattr(dataset_item, "table_info", None)
        if table_info is None:
            return ""
        table_name = str(getattr(table_info, "name", "")).strip()
        column_info_list = getattr(table_info, "column_info_list", []) or []
        column_chunks = []
        for column_info in column_info_list:
            column_name = str(getattr(column_info, "name", "")).strip()
            column_type = str(getattr(column_info, "type", "")).strip()
            if column_name:
                column_chunks.append(f"{column_name} {column_type}".strip())
        if table_name and column_chunks:
            return f"{table_name}({', '.join(column_chunks)})"
        return table_name

    def _extract_success_card_if_enabled(
        self, callback_args: CallbackArguments
    ) -> None:
        if not self.enable_success_cards:
            return
        current_session = callback_args.current_session
        if current_session.task_name != TaskName.DB_BENCH:
            return
        sample_key = str(current_session.sample_index)
        if sample_key in self.success_card_dict:
            return
        if (
            current_session.evaluation_record.outcome
            != SessionEvaluationOutcome.CORRECT
            or current_session.sample_status != SampleStatus.COMPLETED
        ):
            return
        task = callback_args.session_context.task
        getter = getattr(task, "get_dataset_item_for_sample", None)
        if not callable(getter):
            return
        dataset_item = getter(current_session.sample_index)
        question = str(getattr(dataset_item, "instruction", "")).strip()
        skills = list(getattr(dataset_item, "skill_list", []) or [])
        context_anchor = str(
            getattr(getattr(dataset_item, "table_info", None), "name", "")
        ).strip()
        schema_text = self._build_dbbench_schema_text(dataset_item)
        language_model = getattr(callback_args.session_context.agent, "_language_model", None)
        if language_model is None:
            return
        cost_tracker = self._get_cost_tracker(callback_args)
        pre_call_count = len(getattr(cost_tracker, "calls", None) or [])
        rounds_used, tokens_used, runtime_cost = self._get_session_runtime_metadata(
            current_session
        )
        try:
            card = extract_success_card(
                language_model=language_model,
                question=question,
                skills=skills,
                context_anchor=context_anchor,
                schema_text=schema_text,
                runtime_rounds_used=rounds_used,
                runtime_tokens_used=tokens_used,
                runtime_cost=runtime_cost,
                session=current_session,
                agent_role_dict=callback_args.session_context.agent.get_role_dict(),
            )
            if not self._is_valid_success_card(card):
                raise ValueError(
                    "Degenerate success card: missing key_artifact/relevant_context "
                    "or key_artifact does not look like SQL."
                )
            self.success_card_dict[sample_key] = card
            extraction_status = "ok"
            extraction_error: Optional[str] = None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Success-card extraction failed for sample=%s; falling back to raw trace. Error: %s",
                current_session.sample_index,
                e,
            )
            extraction_status = "failed"
            extraction_error = str(e)
        calls = getattr(cost_tracker, "calls", None) or []
        new_calls = calls[pre_call_count:]
        extraction_prompt_tokens = int(sum(call.prompt_tokens for call in new_calls))
        extraction_completion_tokens = int(
            sum(call.completion_tokens for call in new_calls)
        )
        extraction_cost = float(sum(call.total_cost_usd for call in new_calls))
        summary = self.success_card_extraction_summary
        if extraction_status == "ok":
            summary["count"] = int(summary.get("count", 0)) + 1
        summary["total_prompt_tokens"] = int(
            summary.get("total_prompt_tokens", 0)
        ) + extraction_prompt_tokens
        summary["total_completion_tokens"] = int(
            summary.get("total_completion_tokens", 0)
        ) + extraction_completion_tokens
        summary["total_cost_usd"] = float(summary.get("total_cost_usd", 0.0)) + extraction_cost
        summary.setdefault("by_sample", {})[sample_key] = {
            "status": extraction_status,
            "prompt_tokens": extraction_prompt_tokens,
            "completion_tokens": extraction_completion_tokens,
            "cost_usd": extraction_cost,
            "error": extraction_error,
        }

    @staticmethod
    def _is_valid_success_card(card: SuccessCard) -> bool:
        prompt_card = card.prompt_card
        if not prompt_card.relevant_context.strip():
            return False
        key_artifact = prompt_card.key_artifact.strip()
        if not key_artifact:
            return False
        sql_keywords = ("select", "with", "insert", "update", "delete")
        return any(keyword in key_artifact.lower() for keyword in sql_keywords)

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        compact = " ".join(str(text).split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 12].rstrip() + " ...[truncated]"

    def _trim_user_observation(self, content: str) -> str:
        text = str(content or "").strip()
        if not text:
            return text
        if text == "[]":
            return text
        try:
            parsed = ast.literal_eval(text)
        except Exception:  # noqa: BLE001
            return self._truncate_text(text, self.trimmed_result_char_limit)
        if isinstance(parsed, list):
            if len(parsed) <= self.trimmed_result_row_limit:
                return repr(parsed)
            head = parsed[: self.trimmed_result_row_limit]
            remaining = len(parsed) - self.trimmed_result_row_limit
            return f"{repr(head)} ... ({remaining} more rows)"
        return self._truncate_text(text, self.trimmed_result_char_limit)

    def _render_trimmed_trace_block(
        self, session: Session, agent_role_dict: Mapping[Role, str]
    ) -> str:
        try:
            question = session.chat_history.get_item_deep_copy(2).content
        except Exception:  # noqa: BLE001
            question = ""
        lines: list[str] = [f"Question {question}:"]
        for item_index in range(3, session.chat_history.get_value_length()):
            item = session.chat_history.get_item_deep_copy(item_index)
            role_str = agent_role_dict[item.role]
            content = str(item.content or "")
            if item.role == Role.USER:
                content = self._trim_user_observation(content)
            lines.append(f"{role_str}: {content}")
        return "\n".join(lines) + "\n"

    def _render_session_block(
        self, session: Session, agent_role_dict: Mapping[Role, str]
    ) -> str:
        if self.replay_render_mode == "success_card":
            success_card = self.success_card_dict.get(str(session.sample_index))
            if success_card is not None:
                return success_card.render_prompt_card()
        elif self.replay_render_mode == "trimmed_trace":
            return self._render_trimmed_trace_block(session, agent_role_dict)
        try:
            question = session.chat_history.get_item_deep_copy(2).content
        except Exception:  # noqa: BLE001
            question = ""
        session_str = f"Question {question}:\n"
        session_str += session.chat_history.get_value_str(
            agent_role_dict, start_index=3, end_index=None
        )
        return session_str

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        # The session is just created, so its chat_history should be empty.
        assert callback_args.current_session.chat_history.get_value_length() == 0

        # CoPAL-L replay gate: the bandit writes _copal_replay_mode onto the
        # task object before the session is created.  Supported values:
        #   "full"       – inject full trajectories (default, existing behaviour)
        #   "none"       – skip injection, strip placeholder (saves all replay tokens)
        #   "compressed" – handled by LLMLinguaPreviousSampleUtilizationCallback;
        #                  plain callback falls back to "full" if it sees this value.
        task = callback_args.session_context.task
        replay_mode: str = getattr(task, "_copal_replay_mode", "full")

        if replay_mode == "none":
            # Remove the placeholder without injecting anything.
            first_user_prompt = self.original_first_user_prompt.replace(
                self.pattern, ""
            )
            task.chat_history_item_factory.set(0, Role.USER, first_user_prompt)
            return

        # Step1. Construct example_text  ("full" or "compressed" fallback).
        agent_role_dict = callback_args.session_context.agent.get_role_dict()
        example_text = "\n"
        for i, session in enumerate(self.utilized_session_list):
            example_text += self._render_session_block(session, agent_role_dict)
        # Step2. Replace the pattern with the example_text.
        first_user_prompt = self.original_first_user_prompt.replace(
            self.pattern, example_text
        )
        task.chat_history_item_factory.set(0, Role.USER, first_user_prompt)

    def on_agent_inference(self, callback_args: CallbackArguments) -> None:
        last_chat_history_item = (
            callback_args.current_session.chat_history.get_item_deep_copy(-1)
        )
        assert last_chat_history_item.role == Role.AGENT
        last_agent_response = last_chat_history_item.content
        counterfeit_user_response_location = last_agent_response.find("\nuser: ")
        if counterfeit_user_response_location != -1:
            last_agent_response = last_agent_response[
                :counterfeit_user_response_location
            ]
        callback_args.current_session.chat_history.set(
            -1,
            ChatHistoryItem(
                role=Role.AGENT,
                content=last_agent_response,
            ),
        )

    def on_state_save(self, callback_args: CallbackArguments) -> None:
        self._extract_success_card_if_enabled(callback_args)
        json.dump(
            [s.model_dump() for s in self.utilized_session_list],
            open(self._get_utilized_session_list_state_path(), "w"),  # noqa
            indent=2,
        )
        json.dump(
            {
                sample_index: card.model_dump()
                for sample_index, card in self.success_card_dict.items()
            },
            open(self._get_success_card_state_path(), "w"),  # noqa
            indent=2,
        )
        json.dump(
            self.success_card_extraction_summary,
            open(self._get_success_card_summary_state_path(), "w"),  # noqa
            indent=2,
        )
