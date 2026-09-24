"""
LLMLingua-compressed experience replay callback.

Wraps PreviousSampleUtilizationCallback and compresses the injected replay
trajectories using LLMLingua-2 (default, CPU-friendly) or LongLLMLingua
(question-aware, requires a larger local LM) before they are inserted into
the prompt.

Install dependency:  pip install llmlingua
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from src.callbacks.callback import CallbackArguments
from src.typings import Role

from .previous_sample_utilization_callback import PreviousSampleUtilizationCallback

logger = logging.getLogger(__name__)


class LLMLinguaPreviousSampleUtilizationCallback(PreviousSampleUtilizationCallback):
    """
    Identical to PreviousSampleUtilizationCallback except that the replay
    trajectories are compressed by LLMLingua before being injected.

    Parameters
    ----------
    original_first_user_prompt : str
        Injected automatically by CallbackConstructor.
    utilized_sample_count : int
        How many past sessions to keep in the replay buffer.
    compression_rate : float
        Target fraction of tokens to *keep* (e.g. 0.5 keeps 50 %).
        Ignored when target_token > 0.
    target_token : int
        Hard token budget for the compressed replay block.
        Set to -1 to use compression_rate instead.
    compressor_model_name : str
        HuggingFace model ID passed to llmlingua.PromptCompressor.
        Defaults to the LLMLingua-2 XLM-RoBERTa model (~560 MB, CPU-friendly).
    use_llmlingua2 : bool
        True  → LLMLingua-2 (fast, CPU, token-classification model).
        False → LongLLMLingua (question-aware, needs a full causal LM).
    device : str
        "cpu" or "cuda".  LLMLingua-2 works fine on CPU.
    """

    def __init__(
        self,
        original_first_user_prompt: str,
        utilized_sample_count: int,
        compression_rate: float = 0.5,
        target_token: int = -1,
        compressor_model_name: str = (
            "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        ),
        use_llmlingua2: bool = True,
        device: str = "cpu",
        force_compression: bool = False,
    ) -> None:
        super().__init__(
            original_first_user_prompt=original_first_user_prompt,
            utilized_sample_count=utilized_sample_count,
        )
        self.compression_rate = float(compression_rate)
        self.target_token = int(target_token)
        self.compressor_model_name = compressor_model_name
        self.use_llmlingua2 = bool(use_llmlingua2)
        self.device = device
        self.force_compression = bool(force_compression)
        self._compressor: Optional[object] = None  # lazy-loaded

    @staticmethod
    def _extract_current_question(task: Any, sample_index: Any) -> str:
        """
        Best-effort extraction of the *actual* current sample instruction/question.

        LongLLMLingua is question-aware, so conditioning it on the generic system
        prompt is much weaker than conditioning it on the active DBBench query.
        """
        dataset = getattr(task, "dataset", None)
        dataset_item: Any = None
        if isinstance(dataset, dict) and sample_index in dataset:
            dataset_item = dataset[sample_index]
        else:
            getter = getattr(task, "get_dataset_item_for_sample", None)
            if not callable(getter):
                getter = getattr(task, "get_data", None)
            if callable(getter):
                try:
                    dataset_item = getter(sample_index)
                except Exception:  # noqa: BLE001
                    dataset_item = None
        if dataset_item is not None:
            for field_name in ("instruction", "question"):
                value = getattr(dataset_item, field_name, None)
                if value:
                    return str(value)
            try:
                return str(dataset_item.model_dump())
            except Exception:  # noqa: BLE001
                return str(dataset_item)
        return ""

    @staticmethod
    def _compression_looks_unsafe(original_text: str, compressed_text: str) -> bool:
        """
        Replay trajectories act like few-shot demonstrations, not passive docs.
        If compression strips key protocol markers, the agent often starts
        failing validation. In that case, fall back to the original replay.
        """
        if not compressed_text.strip():
            return True
        critical_markers = (
            "Question ",
            "Action: Operation",
            "```sql",
            "Action: Answer",
            "Final Answer:",
        )
        for marker in critical_markers:
            if marker in original_text and marker not in compressed_text:
                return True
        if len(compressed_text.strip()) < max(64, int(0.15 * len(original_text.strip()))):
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_compressor(self) -> object:
        """Lazy-load the LLMLingua compressor on first use."""
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
                "LLMLinguaPreviousSampleUtilizationCallback: loaded compressor "
                "model=%s use_llmlingua2=%s device=%s",
                self.compressor_model_name,
                self.use_llmlingua2,
                self.device,
            )
        return self._compressor

    def _compress(
        self,
        context_blocks: Sequence[str],
        question_context: str,
        original_text: str,
    ) -> str:
        """
        Compress replay trajectories and return the compressed string.
        Falls back to the original text if compression raises an exception.
        """
        if not original_text.strip():
            return original_text
        try:
            compressor = self._load_compressor()
            kwargs: dict = dict(
                context=list(context_blocks),
            )
            if self.target_token > 0:
                kwargs["target_token"] = self.target_token
            else:
                kwargs["rate"] = self.compression_rate

            if not self.use_llmlingua2:
                # LongLLMLingua: pass the *actual current sample question* so
                # replay compression is conditioned on the query we are solving.
                kwargs["question"] = question_context
                kwargs["rank_method"] = "longllmlingua"
                kwargs["condition_compare"] = True
                kwargs["reorder_context"] = "sort"

            result = compressor.compress_prompt(**kwargs)  # type: ignore[union-attr]
            compressed: str = result.get("compressed_prompt", original_text)
            original_tokens: int = result.get("origin_tokens", 0)
            compressed_tokens: int = result.get("compressed_tokens", 0)
            logger.info(
                "LLMLingua compression: %d → %d tokens (%.1f %%)",
                original_tokens,
                compressed_tokens,
                100.0 * compressed_tokens / max(original_tokens, 1),
            )
            if self._compression_looks_unsafe(original_text, compressed):
                logger.warning(
                    "LLMLingua compression removed critical replay structure; "
                    "falling back to uncompressed replay."
                )
                return original_text
            return compressed
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "LLMLingua compression failed, using original text. Error: %s", exc
            )
            return original_text

    # ------------------------------------------------------------------
    # Override on_session_create to inject *compressed* replay content
    # ------------------------------------------------------------------

    def on_session_create(self, callback_args: CallbackArguments) -> None:
        assert callback_args.current_session.chat_history.get_value_length() == 0

        # CoPAL-L replay gate.
        task = callback_args.session_context.task
        replay_mode: str = (
            "compressed"
            if self.force_compression
            else getattr(task, "_copal_replay_mode", "full")
        )
        if replay_mode == "none":
            super().on_session_create(callback_args)
            return
        if replay_mode == "full":
            # Important: in replay-gated mode, "full" must really mean
            # *uncompressed* replay. Delegate to the parent implementation.
            super().on_session_create(callback_args)
            return

        # Nothing to compress if the replay buffer is empty.
        if not self.utilized_session_list:
            super().on_session_create(callback_args)
            return

        # Build replay trajectories as separate context blocks so LongLLMLingua
        # can score/allocate compression across demonstrations instead of
        # flattening everything into one undifferentiated blob.
        agent_role_dict = callback_args.session_context.agent.get_role_dict()
        context_blocks: list[str] = []
        for session in self.utilized_session_list:
            try:
                question = session.chat_history.get_item_deep_copy(2).content
            except Exception:  # noqa: BLE001
                question = ""
            session_str = f"Question {question}:\n"
            session_str += session.chat_history.get_value_str(
                agent_role_dict, start_index=3, end_index=None
            )
            context_blocks.append(session_str)
        example_text = "\n" + "\n".join(context_blocks)

        # Use the actual active DBBench instruction as question context.
        question_context = self._extract_current_question(
            task=task,
            sample_index=callback_args.current_session.sample_index,
        )
        if not question_context:
            question_context = self.original_first_user_prompt.replace(
                self.pattern, ""
            ).strip()

        # --- Compress ---
        compressed_example_text = self._compress(
            context_blocks=context_blocks,
            question_context=question_context,
            original_text=example_text,
        )

        # --- Inject into prompt ---
        first_user_prompt = self.original_first_user_prompt.replace(
            self.pattern, compressed_example_text
        )
        callback_args.session_context.task.chat_history_item_factory.set(
            0, Role.USER, first_user_prompt
        )
