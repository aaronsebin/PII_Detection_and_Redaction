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


def _predict_spans(client: OpenAI, task_id: str, document_text: str) -> list[RedactionSpan]:
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
        model="gpt-4o-mini",
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


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the baseline")

    client = OpenAI(api_key=api_key)
    env = PIIRedactionEnv()
    task_scores: dict[str, float] = {}

    for task_id in TASK_ORDER:
        observation = env.reset(seed=42, task_id=task_id)
        predicted_spans = _predict_spans(client, task_id, observation.document_text)
        result = env.step(PIIAction(spans=predicted_spans, submit=True))
        score = float(result.final_score or 0.0)
        task_scores[task_id] = score
        print(f"{task_id}: {score:.4f}")

    print(f"mean: {mean(task_scores.values()):.4f}")
    env.close()


if __name__ == "__main__":
    main()
