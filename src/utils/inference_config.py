from typing import Any


def set_generation_token_budget(agent: Any, token_budget: int) -> None:
    """
    Route token-budget overrides to the right inference-config field for the
    active backend.

    Local Hugging Face models use `max_new_tokens`. OpenAI chat models use
    `max_completion_tokens`.
    """
    if not hasattr(agent, "_inference_config_dict"):
        return
    if getattr(agent, "_inference_config_dict") is None:
        setattr(agent, "_inference_config_dict", {})
    inference_config = agent._inference_config_dict  # type: ignore[attr-defined]
    language_model = getattr(agent, "_language_model", None)
    model_class_name = language_model.__class__.__name__.lower()

    if (
        "max_completion_tokens" in inference_config
        or model_class_name == "openailanguagemodel"
    ):
        inference_config["max_completion_tokens"] = int(token_budget)
        inference_config.pop("max_new_tokens", None)
    else:
        inference_config["max_new_tokens"] = int(token_budget)
        inference_config.pop("max_completion_tokens", None)
