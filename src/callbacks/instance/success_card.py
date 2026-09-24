from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel

from src.typings import ChatHistory, ChatHistoryItem, Role, Session


class RetrievalMetadata(BaseModel):
    task_pattern: str
    skills: list[str]
    context_anchor: str
    question_raw: str
    question_template: Optional[str] = None


class PromptCard(BaseModel):
    original_task: str
    relevant_context: str
    successful_procedure: str = ""
    key_artifact: str
    observed_result_example: Optional[str] = None
    final_answer_example: Optional[str] = None
    why_it_worked: str
    answer_format: str
    common_pitfall: Optional[str] = None
    correction_snippet: Optional[str] = None


class RuntimeMetadata(BaseModel):
    rounds_used: Optional[int] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class SuccessCard(BaseModel):
    retrieval: RetrievalMetadata
    prompt_card: PromptCard
    runtime: RuntimeMetadata

    def render_prompt_card(self) -> str:
        lines = [
            "Verified Successful Reference Example",
            "Use this as a pattern, not as an exact answer.",
        ]
        prompt_card = self.prompt_card
        if prompt_card.original_task.strip():
            lines.append(f"Original Task: {prompt_card.original_task.strip()}")
        if prompt_card.relevant_context.strip():
            lines.append(f"Relevant Context: {prompt_card.relevant_context.strip()}")
        if prompt_card.successful_procedure.strip():
            lines.append(
                f"Successful Procedure: {prompt_card.successful_procedure.strip()}"
            )
        lines.append("Action: Operation")
        lines.append("```sql")
        lines.append(prompt_card.key_artifact.strip())
        lines.append("```")
        if prompt_card.observed_result_example is not None:
            value = str(prompt_card.observed_result_example).strip()
            if value:
                lines.append(f"Observed Result Example: {value}")
        if prompt_card.correction_snippet is not None:
            value = str(prompt_card.correction_snippet).strip()
            if value:
                lines.append(f"Correction Note: {value}")
        if prompt_card.final_answer_example is not None:
            value = str(prompt_card.final_answer_example).strip()
            if value:
                lines.append("Action: Answer")
                lines.append(f"Final Answer: {value}")
        if prompt_card.why_it_worked.strip():
            lines.append(f"Why It Worked: {prompt_card.why_it_worked.strip()}")
        if prompt_card.answer_format.strip():
            lines.append(f"Answer Format: {prompt_card.answer_format.strip()}")
        if prompt_card.common_pitfall is not None:
            value = str(prompt_card.common_pitfall).strip()
            if value:
                lines.append(f"Common Pitfall: {value}")
        return "\n".join(lines) + "\n"


EXTRACTION_SYSTEM_PROMPT = """You convert successful DBBench SQL agent trajectories into compact reusable hybrid reference cards.

Return ONLY valid JSON. Do not use markdown fences. Do not explain your answer.

Your job:
- produce a compact, reusable reference card for a FUTURE model
- preserve the task-to-SQL mapping by keeping the explanation concrete
- prefer concise, concrete notes over abstract summaries
- if there was no meaningful pitfall, set common_pitfall to an empty string
- if there was no meaningful correction sequence, set correction_snippet to an empty string
- if the task was solved in one direct SQL step, successful_procedure may be empty

The JSON schema is:
{
  "task_pattern": "short abstract task type",
  "question_template": "optional generalized question form",
  "successful_procedure": "short generalized action pattern",
  "why_it_worked": "key principle or decision",
  "answer_format": "how the final answer should be presented",
  "common_pitfall": "optional anti-pattern from failed intermediate attempts",
  "correction_snippet": "optional 1-2 sentence note about what changed between a bad attempt and the final SQL"
}

Keep the card concise and concrete. Do not repeat the full task verbatim."""


def _extract_json_object(text: str) -> Mapping[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in extraction output.")
    return json.loads(text[start : end + 1])


def build_dbbench_extraction_prompt(
    *,
    session: Session,
    agent_role_dict: Mapping[Role, str],
    question: str,
    skills: Sequence[str],
    context_anchor: str,
    schema_text: str,
) -> str:
    transcript = session.chat_history.get_value_str(
        agent_role_dict, start_index=3, end_index=None
    )
    return f"""Create a compact reusable hybrid reference card from this successful DBBench trajectory.

Known metadata:
- Raw question: {question}
- Skills: {list(skills)}
- Context anchor: {context_anchor}
- Schema: {schema_text}

Successful trajectory:
{transcript}

Return JSON only using the required schema."""


def _truncate_text(text: str, max_chars: int = 240) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 4].rstrip() + " ..."


def _extract_last_sql_from_session(session: Session) -> str:
    for item_index in range(session.chat_history.get_value_length() - 1, -1, -1):
        item = session.chat_history.get_item_deep_copy(item_index)
        if item.role != Role.AGENT:
            continue
        content = str(item.content or "")
        block_match = re.search(
            r"```sql\s*(.*?)\s*```", content, flags=re.IGNORECASE | re.DOTALL
        )
        if block_match:
            return block_match.group(1).strip()
        generic_block_match = re.search(r"```\s*(.*?)\s*```", content, flags=re.DOTALL)
        if generic_block_match and "Action: Operation" in content:
            return generic_block_match.group(1).strip()
    return ""


def _extract_final_answer_example(session: Session) -> str:
    if session.chat_history.get_value_length() == 0:
        return ""
    final_item = session.chat_history.get_item_deep_copy(-1)
    if final_item.role != Role.AGENT:
        return ""
    content = str(final_item.content or "")
    final_answer_match = re.search(r"Final Answer:\s*(.*)", content, flags=re.DOTALL)
    if final_answer_match:
        return _truncate_text(final_answer_match.group(1).strip(), max_chars=240)
    return _truncate_text(content.strip(), max_chars=240)


def _extract_observed_result_example(session: Session) -> str:
    value_length = session.chat_history.get_value_length()
    for item_index in range(value_length - 2, -1, -1):
        item = session.chat_history.get_item_deep_copy(item_index)
        if item.role == Role.USER:
            return _truncate_text(str(item.content or "").strip(), max_chars=240)
    return ""


def extract_success_card(
    *,
    language_model: Any,
    question: str,
    skills: Sequence[str],
    context_anchor: str,
    schema_text: str,
    runtime_rounds_used: Optional[int],
    runtime_tokens_used: Optional[int],
    runtime_cost: Optional[float],
    session: Session,
    agent_role_dict: Mapping[Role, str],
    inference_config_dict: Optional[Mapping[str, Any]] = None,
) -> SuccessCard:
    chat_history = ChatHistory()
    chat_history.inject(
        ChatHistoryItem(
            role=Role.USER,
            content=build_dbbench_extraction_prompt(
                session=session,
                agent_role_dict=agent_role_dict,
                question=question,
                skills=skills,
                context_anchor=context_anchor,
                schema_text=schema_text,
            ),
        )
    )
    if inference_config_dict is None:
        inference_config_dict = {
            "max_new_tokens": 448,
            "do_sample": False,
        }
    output_item = language_model.inference(
        [chat_history],
        inference_config_dict=inference_config_dict,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
    )[0]
    payload = _extract_json_object(output_item.content)
    final_sql = _extract_last_sql_from_session(session)
    if not final_sql:
        final_sql = str(payload.get("key_artifact", "")).strip()
    observed_result_example = _extract_observed_result_example(session)
    final_answer_example = _extract_final_answer_example(session)
    return SuccessCard(
        retrieval=RetrievalMetadata(
            task_pattern=str(payload.get("task_pattern", "")).strip(),
            skills=[str(skill) for skill in skills],
            context_anchor=str(context_anchor).strip(),
            question_raw=str(question).strip(),
            question_template=(
                str(payload.get("question_template", "")).strip() or None
            ),
        ),
        prompt_card=PromptCard(
            original_task=str(question).strip(),
            relevant_context=str(schema_text).strip(),
            successful_procedure=str(
                payload.get("successful_procedure", "")
            ).strip(),
            key_artifact=final_sql,
            observed_result_example=(observed_result_example or None),
            final_answer_example=(final_answer_example or None),
            why_it_worked=str(payload.get("why_it_worked", "")).strip(),
            answer_format=str(payload.get("answer_format", "")).strip(),
            common_pitfall=(str(payload.get("common_pitfall", "")).strip() or None),
            correction_snippet=(
                str(payload.get("correction_snippet", "")).strip() or None
            ),
        ),
        runtime=RuntimeMetadata(
            rounds_used=runtime_rounds_used,
            tokens_used=runtime_tokens_used,
            cost=runtime_cost,
        ),
    )
