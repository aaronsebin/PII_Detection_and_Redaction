from __future__ import annotations

import json
import os
from statistics import mean

from openai import OpenAI

try:
    from .env import PIIRedactionEnv
    from .models import PIIAction, PIIType, RedactionSpan
    from .tasks import TASK_ORDER
except ImportError:
    from env import PIIRedactionEnv
    from models import PIIAction, PIIType, RedactionSpan
    from tasks import TASK_ORDER


def _extract_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    if hasattr(response, "choices") and response.choices:
        return response.choices[0].message.content or ""
    raise ValueError("OpenAI response did not contain text output")


def _predict_spans(
    client: OpenAI,
    model_name: str,
    task_id: str,
    document_text: str,
) -> list[RedactionSpan]:
    schema = {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer"},
                        "end": {"type": "integer"},
                        "pii_type": {
                            "type": "string",
                            "enum": [member.value for member in PIIType],
                        },
                        "text": {"type": "string"},
                    },
                    "required": ["start", "end", "pii_type", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["spans"],
        "additionalProperties": False,
    }
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You identify exact PII spans in synthetic documents. Return JSON only.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Task: {task_id}\n"
                            "Return all PII spans as exact character offsets over the provided document.\n"
                            f"Document:\n{document_text}"
                        ),
                    }
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "pii_spans",
                "schema": schema,
                "strict": True,
            }
        },
    )
    payload = json.loads(_extract_text(response))
    return [RedactionSpan(**span) for span in payload["spans"]]


def _log(prefix: str, payload: dict[str, object]) -> None:
    print(f"{prefix} {json.dumps(payload, ensure_ascii=True)}")


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "")
    hf_token = os.getenv("HF_TOKEN", "")
    missing = [
        name
        for name, value in [
            ("API_BASE_URL", api_base_url),
            ("MODEL_NAME", model_name),
            ("HF_TOKEN", hf_token),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    env = PIIRedactionEnv()
    task_scores: list[float] = []

    try:
        for task_id in TASK_ORDER:
            observation = env.reset(seed=42, task_id=task_id)
            state = env.state()
            _log(
                "[START]",
                {
                    "task_id": observation.task_id,
                    "difficulty": observation.difficulty,
                    "document_id": state.episode_id,
                },
            )

            predicted_spans = _predict_spans(
                client,
                model_name,
                task_id,
                observation.document_text,
            )
            result = env.step(PIIAction(spans=predicted_spans, submit=True))
            score = float(result.final_score or 0.0)
            task_scores.append(score)
            _log(
                "[STEP]",
                {
                    "step": state.step_count,
                    "action_type": "submit",
                    "reward": float(result.reward),
                    "done": bool(result.done),
                },
            )
            _log(
                "[END]",
                {
                    "task_id": observation.task_id,
                    "difficulty": observation.difficulty,
                    "score": score,
                    "steps": state.step_count,
                },
            )
    finally:
        env.close()

    print(
        json.dumps(
            {
                "mean_score": float(mean(task_scores)),
                "tasks_completed": len(task_scores),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
