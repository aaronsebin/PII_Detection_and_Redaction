from __future__ import annotations

import json
import os
import statistics

from openai import OpenAI

try:
    from .env import PIIRedactionEnv
    from .models import PIIAction, PIIType, RedactionSpan
    from .tasks import TASK_ORDER
except ImportError:
    from env import PIIRedactionEnv
    from models import PIIAction, PIIType, RedactionSpan
    from tasks import TASK_ORDER


def log_start(task, env, model): print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step, action, reward, done, error): print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)
def log_end(success, steps, score, rewards): print(f"[END] success={str(success).lower()} steps={steps} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


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
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You identify exact PII spans in synthetic documents. "
                    "Return JSON only with schema: "
                    f"{json.dumps(schema, ensure_ascii=True)}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task: {task_id}\n"
                    "Return all PII spans as exact character offsets over the provided document.\n"
                    f"Document:\n{document_text}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI chat completion did not contain message content")
    payload = json.loads(content)
    valid_spans = [s for s in payload["spans"] if s.get("end", 0) > s.get("start", 0)]
    return [RedactionSpan(**span) for span in valid_spans]


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    hf_token = os.getenv("HF_TOKEN")
    missing = [
        name
        for name, value in [
            ("HF_TOKEN", hf_token),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    env = PIIRedactionEnv()
    all_scores: list[float] = []

    try:
        for task_id in TASK_ORDER:
            score = 0.0
            steps = 0
            rewards: list[float] = []
            log_start(task_id, "pii_redaction_env", model_name)
            try:
                observation = env.reset(seed=42, task_id=task_id)
                predicted_spans = _predict_spans(
                    client,
                    model_name,
                    task_id,
                    observation.document_text,
                )
                action = PIIAction(spans=predicted_spans, submit=True)
                result = env.step(action)
                reward = float(result.reward)
                rewards.append(reward)
                steps = env.state().step_count
                score = float(result.final_score or 0.0)
                all_scores.append(score)
                action_str = f"submit({len(predicted_spans)}_spans)"
                log_step(steps, action_str, reward, bool(result.done), None)
            finally:
                log_end(score >= 0.1, steps, score, rewards)
        print(json.dumps({"mean_score": round(statistics.mean(all_scores), 4) if all_scores else 0.0, "tasks_completed": len(all_scores)}, ensure_ascii=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
