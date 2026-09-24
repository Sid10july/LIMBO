from __future__ import annotations

import logging
from typing import Optional, Sequence

from src.callbacks.callback import CallbackArguments
from src.typings import Role

from .llmlingua_previous_sample_utilization_callback import (
    LLMLinguaPreviousSampleUtilizationCallback,
)
from .retrieval_previous_sample_utilization_callback import (
    RetrievalPreviousSampleUtilizationCallback,
)

logger = logging.getLogger(__name__)


class RetrievalLLMLinguaPreviousSampleUtilizationCallback(
    RetrievalPreviousSampleUtilizationCallback
):
    """
    Retrieval-based replay followed by LLMLingua/LongLLMLingua compression.

    This callback first selects relevant successful trajectories from a larger
    memory bank (via RetrievalPreviousSampleUtilizationCallback), then compresses
    the selected trajectories before injecting them into the prompt.

    Replay-mode behavior:
    - "none":       no replay
    - "full":       recent full replay, uncompressed
    - "retrieved":  retrieved replay, uncompressed
    - "compressed": retrieved replay, compressed
    """

    def __init__(
        self,
        original_first_user_prompt: str,
        utilized_sample_count: int,
        retrieved_sample_count: int = 16,
        full_replay_sample_count: int = 16,
        lexical_weight: float = 0.35,
        skill_weight: float = 0.50,
        table_weight: float = 0.15,
        compression_rate: float = 0.8,
        target_token: int = -1,
        compressor_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        use_llmlingua2: bool = False,
        device: str = "cuda",
    ) -> None:
        super().__init__(
            original_first_user_prompt=original_first_user_prompt,
            utilized_sample_count=utilized_sample_count,
            retrieved_sample_count=retrieved_sample_count,
            full_replay_sample_count=full_replay_sample_count,
            lexical_weight=lexical_weight,
            skill_weight=skill_weight,
            table_weight=table_weight,
        )
        self.compression_rate = float(compression_rate)
        self.target_token = int(target_token)
        self.compressor_model_name = compressor_model_name
        self.use_llmlingua2 = bool(use_llmlingua2)
        self.device = device
        self._compressor: Optional[object] = None

    def _load_compressor(self) -> object:
        if self._compressor is None:
            try:
                from llmlingua import PromptCompressor  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "llmlingua is not installed. Run: pip install llmlingua"
                ) from exc
            self._compressor = PromptCompressor(
                model_name=self.compressor_model_name,
                use_llmlingua2=self.use_llmlingua2,
                device_map=self.device,
            )
            logger.info(
                "RetrievalLLMLingua callback: loaded compressor model=%s "
                "use_llmlingua2=%s device=%s",
                self.compressor_model_name,
                self.use_llmlingua2,
                self.device,
            )
        return self._compressor

    def _compress(
        self,
        *,
        context_blocks: Sequence[str],
        question_context: str,
        original_text: str,
    ) -> str:
        if not original_text.strip():
            return original_text
        try:
            compressor = self._load_compressor()
            kwargs: dict = {"context": list(context_blocks)}
            if self.target_token > 0:
                kwargs["target_token"] = self.target_token
            else:
                kwargs["rate"] = self.compression_rate
            if not self.use_llmlingua2:
                kwargs["question"] = question_context
                kwargs["rank_method"] = "longllmlingua"
                kwargs["condition_compare"] = True
                kwargs["reorder_context"] = "sort"
            result = compressor.compress_prompt(**kwargs)  # type: ignore[union-attr]
            compressed: str = result.get("compressed_prompt", original_text)
            original_tokens: int = result.get("origin_tokens", 0)
            compressed_tokens: int = result.get("compressed_tokens", 0)
            logger.info(
                "Retrieval+LLMLingua compression: %d -> %d tokens (%.1f %%)",
                original_tokens,
                compressed_tokens,
                100.0 * compressed_tokens / max(original_tokens, 1),
            )
            if LLMLinguaPreviousSampleUtilizationCallback._compression_looks_unsafe(
                original_text, compressed
            ):
                logger.warning(
                    "Retrieval+LLMLingua compression removed critical replay "
                    "structure; falling back to uncompressed retrieved replay."
                )
                return original_text
            return compressed
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Retrieval+LLMLingua compression failed; using uncompressed "
                "retrieved replay. Error: %s",
                exc,
            )
            return original_text

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        assert callback_args.current_session.chat_history.get_value_length() == 0

        task = callback_args.session_context.task
        replay_mode: str = getattr(task, "_copal_replay_mode", "compressed")
        replay_sample_count_override = getattr(
            task, "_copal_replay_sample_count", None
        )

        # Keep "none", "full", and "retrieved" behavior identical to the
        # retrieval baseline so this callback can later be used in gated runs.
        if replay_mode in {"none", "full", "retrieved"}:
            super().on_session_create(callback_args)
            return

        if not self.utilized_session_list:
            super().on_session_create(callback_args)
            return

        selected_session_list = self._select_retrieved_sessions(
            task=task,
            current_sample_index=callback_args.current_session.sample_index,
            replay_sample_count=replay_sample_count_override,
        )
        if len(selected_session_list) == 0:
            super().on_session_create(callback_args)
            return

        agent_role_dict = callback_args.session_context.agent.get_role_dict()
        context_blocks = [
            self._render_session_block(session, agent_role_dict)
            for session in selected_session_list
        ]
        example_text = "\n" + "\n".join(context_blocks)

        question_context = (
            LLMLinguaPreviousSampleUtilizationCallback._extract_current_question(
                task=task,
                sample_index=callback_args.current_session.sample_index,
            )
        )
        if not question_context:
            question_context = self.original_first_user_prompt.replace(
                self.pattern, ""
            ).strip()

        compressed_example_text = self._compress(
            context_blocks=context_blocks,
            question_context=question_context,
            original_text=example_text,
        )
        first_user_prompt = self.original_first_user_prompt.replace(
            self.pattern, compressed_example_text
        )
        task.chat_history_item_factory.set(0, Role.USER, first_user_prompt)
