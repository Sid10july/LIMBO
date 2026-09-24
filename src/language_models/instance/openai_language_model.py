from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import openai
import os
from typing import Any, Optional, Sequence, Mapping, TypeGuard

from src.language_models.language_model import LanguageModel
from src.typings import (
    Role,
    ChatHistoryItem,
    LanguageModelContextLimitException,
    ChatHistory,
)
from src.utils import RetryHandler, ExponentialBackoffStrategy


class OpenaiLanguageModel(LanguageModel):
    """
    To keep the name of the class consistent with the name of file, use OpenaiAgent instead of OpenAIAgent.
    """

    def __init__(
        self,
        model_name: str,
        role_dict: Mapping[str, str],
        api_key: Optional[str] = None,
        api_key_env_var: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        maximum_prompt_token_count: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """
        max_prompt_tokens: The maximum number of tokens that can be used in the prompt. It can be used to set the
            context limit manually. If it is set to None, the context limit will be the same as the context length of
            the model selected.
        """
        super().__init__(role_dict)
        self.model_name = model_name
        if not api_key:
            api_key = os.environ.get(api_key_env_var)
        if not base_url:
            base_url = os.environ.get("OPENAI_BASE_URL")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        self.client = OpenAI(**client_kwargs)
        self.maximum_prompt_token_count = maximum_prompt_token_count
        self.last_input_token_count: Optional[int] = None
        self.last_output_token_count: Optional[int] = None

    @staticmethod
    def _is_valid_message_list(
        message_list: list[Mapping[str, str]],
    ) -> TypeGuard[list[ChatCompletionMessageParam]]:
        for message_dict in message_list:
            if (
                "role" not in message_dict.keys()
                or "content" not in message_dict.keys()
            ):
                return False
        return True

    @RetryHandler.handle(
        max_retries=3,
        retry_on=(
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
        ),
        waiting_strategy=ExponentialBackoffStrategy(interval=(None, 60), multiplier=2),
    )
    def _get_completion_content(
        self,
        message_list: Sequence[ChatCompletionMessageParam],
        inference_config_dict: Mapping[str, Any],
    ) -> Sequence[str]:
        """
        I do not know what will happen when the context limit is reached. According to OpenAI documents, there is no
        type of error for the context limit. So I guess the model will return an empty response when the context limit
        is reached. This may be a potential bug and I apologize in advance.
        There are also some issues on GitHub state that the model will raise openai.BadRequestError in this situation.
        So I also handle this error in the code.
        Reference:
        https://platform.openai.com/docs/guides/error-codes#python-library-error-types
        https://github.com/run-llama/llama_index/discussions/11889
        """
        request_config = dict(inference_config_dict)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_list,
                **request_config,
            )
        except openai.BadRequestError as e:
            if "context length" in str(e):
                # Raise LanguageModelContextLimitException to skip retrying.
                raise LanguageModelContextLimitException(
                    f"Model {self.model_name} reaches the context limit. "
                )
            if "max_completion_tokens" in request_config:
                # Some OpenAI-compatible Chat Completions endpoints only
                # accept the legacy max_tokens spelling. Retry once with the
                # equivalent field, then surface the provider response.
                request_config["max_tokens"] = request_config.pop(
                    "max_completion_tokens"
                )
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message_list,
                    **request_config,
                )
            else:
                # Invalid requests are not transient; surface the provider's
                # error immediately instead of repeating a doomed request.
                raise e
        usage = completion.usage
        self.last_input_token_count = (
            int(usage.prompt_tokens) if usage is not None else None
        )
        self.last_output_token_count = (
            int(usage.completion_tokens) if usage is not None else None
        )
        if (
            usage is not None
            and self.maximum_prompt_token_count is not None
            and usage.prompt_tokens > self.maximum_prompt_token_count
        ):
            raise LanguageModelContextLimitException(
                f"Model {self.model_name} reaches the context limit. "
                f"Current prompt tokens: {usage.prompt_tokens}. "
                f"Max prompt tokens: {self.maximum_prompt_token_count}."
            )
        cost_tracker = getattr(self, "cost_tracker", None)
        if (
            cost_tracker is not None
            and self.last_input_token_count is not None
            and self.last_output_token_count is not None
        ):
            try:
                cost_tracker.log_call(
                    self.model_name,
                    self.last_input_token_count,
                    self.last_output_token_count,
                )
            except Exception:
                pass
        content_list: list[str] = []
        content_all_invalid_flag: bool = True
        for choice in completion.choices:
            content = choice.message.content
            if content is not None and len(content) > 0:
                content_all_invalid_flag = False
            content_list.append(content or "")
        if content_all_invalid_flag:
            raise LanguageModelContextLimitException(
                f"Model {self.model_name} returns empty response. The context limit may be reached."
            )
        return content_list

    def _inference(
        self,
        batch_chat_history: Sequence[ChatHistory],
        inference_config_dict: Mapping[str, Any],
        system_prompt: str,
    ) -> Sequence[ChatHistoryItem]:
        """
        system_prompt: It is usually called as system_prompt. But in OpenAI documents, it is called as developer_prompt.
            But in practice, using `message_list = [{"role": "developer", "content": self.system_prompt}]` will raise an
            error. So all after all, I call it as system_prompt.
            Reference:
            https://platform.openai.com/docs/guides/text-generation#messages-and-roles
            https://platform.openai.com/docs/api-reference/chat/create
        inference_config_dict: Other config for OpenAI().chat.completions.create.
            e.g.:
            max_completion_tokens: The maximum number of tokens that can be generated in the chat completion. Notice
                that max_tokens is deprecated.
            Reference:
            https://platform.openai.com/docs/api-reference/chat/create#chat-create-max_completion_tokens
        """
        # region Construct batch_message_list
        message_list_prefix: list[ChatCompletionMessageParam]
        if len(system_prompt) > 0:
            message_list_prefix = [{"role": "system", "content": system_prompt}]
        else:
            message_list_prefix = []
        batch_message_list: list[Sequence[ChatCompletionMessageParam]] = []
        for chat_history in batch_chat_history:
            conversion_result = self._convert_chat_history_to_message_list(chat_history)
            assert OpenaiLanguageModel._is_valid_message_list(conversion_result)
            batch_message_list.append(message_list_prefix + conversion_result)
        # endregion
        # region Generate output
        output_str_list: list[str] = []
        for message_list in batch_message_list:
            output_str_list.extend(
                self._get_completion_content(message_list, inference_config_dict)
            )
        # endregion
        # region Convert output to ChatHistoryItem
        return [
            ChatHistoryItem(role=Role.AGENT, content=output_str)
            for output_str in output_str_list
        ]
        # endregion
