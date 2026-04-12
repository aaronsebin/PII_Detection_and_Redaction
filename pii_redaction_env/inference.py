from __future__ import annotations

import json
import os
import statistics
import sys

from openai import OpenAI

try:
    from .env import PIIRedactionEnv
    from .models import PIIAction, PIIType, RedactionSpan
    from .tasks import TASK_ORDER
except ImportError:
    from env import PIIRedactionEnv
    from models import PIIAction, PIIType, RedactionSpan
    from tasks import TASK_ORDER

# Use 0.01 as minimum — 1e-6 rounds to 0.00 with :.2f which fails validation
MIN_SCORE = 0.01
MAX_SCORE = 0.99


def log_start(task, env, model): print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step, action, reward, done, error): print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)
def log_end(success, steps, score, rewards): print(f"[END] success={str(success).lower()} steps={steps} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)


def _clamp(score: float) -> float:
    """Ensure score is strictly within (0.01, 0.99) — never rounds to 0.00 or 1.00."""
    return max(MIN_SCORE, min(MAX_SCORE, float(score)))


PII_TYPE_ALIASES = {
    "name": "PERSON",
    "person": "PERSON",
    "full_name": "PERSON",
    "email": "EMAIL",
    "email_address": "EMAIL",
    "phone": "PHONE",
    "phone_number": "PHONE",
    "telephone": "PHONE",
    "ssn": "SSN",
    "social_security": "SSN",
    "social_security_number": "SSN",
    "address": "ADDRESS",
    "street_address": "ADDRESS",
    "dob": "DATE_OF_BIRTH",
    "date_of_birth": "DATE_OF_BIRTH",
    "birthday": "DATE_OF_BIRTH",
    "credit_card": "CREDIT_CARD",
    "credit_card_number": "CREDIT_CARD",
    "ip": "IP_ADDRESS",
    "ip_address": "IP_ADDRESS",
    "passport": "PASSPORT",
    "passport_number": "PASSPORT",
    "quasi": "QUASI_IDENTIFIER",
    "quasi_identifier": "QUASI_IDENTIFIER",
}


def _normalize_pii_type(raw_type: str) -> str | None:
    normalized = raw_type.strip().upper()
    valid = {member.value for member in PIIType}
    if normalized in valid:
        return normalized
    alias = PII_TYPE_ALIASES.get(raw_type.strip().lower())
    return alias if alias else None


def _fix_span_offsets(
    spans: list[RedactionSpan],
    document_text: str,
) -> list[RedactionSpan]:
    fixed = []
    for span in spans:
        text = span.text
        if not text:
            continue
        actual_start = document_text.find(text)
        if actual_start == -1:
            lower_doc = document_text.lower()
            lower_text = text.lower()
            actual_start = lower_doc.find(lower_text)
            if actual_start == -1:
                continue
        actual_end = actual_start + len(text)
        fixed.append(RedactionSpan(
            start=actual_start,
            end=actual_end,
            pii_type=span.pii_type,
            text=text,
        ))
    return fixed


def _predict_spans(
    client: OpenAI,
    model_name: str,
    task_id: str,
    document_text: str,
) -> list[RedactionSpan]:
    """Call the LLM to predict PII spans. Returns empty list on any failure."""
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
    try:
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
            print(f"[WARN] Empty response from model for task={task_id}", file=sys.stderr, flush=True)
            return []
        # Strip any extra data after the first complete JSON object
        content = content.strip()
        brace_count = 0
        end_idx = 0
        for i, char in enumerate(content):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        if end_idx > 0:
            content = content[:end_idx]
        payload = json.loads(content)
    except Exception as exc:
        print(f"[WARN] LLM call failed for task={task_id}: {exc}", file=sys.stderr, flush=True)
        return []

    result_spans = []
    try:
        for s in payload.get("spans", []):
            raw_type = s.get("pii_type", "")
            normalized_type = _normalize_pii_type(raw_type)
            if not normalized_type:
                continue
            text = s.get("text", "")
            if not text:
                continue
            result_spans.append(RedactionSpan(
                start=s.get("start", 0),
                end=s.get("end", 0),
                pii_type=normalized_type,
                text=text,
            ))
    except Exception as exc:
        print(f"[WARN] Span parsing failed for task={task_id}: {exc}", file=sys.stderr, flush=True)
        return []

    return result_spans


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    all_scores: list[float] = []

    for task_id in TASK_ORDER:
        # Fresh env per task — avoids state leakage between tasks
        env = PIIRedactionEnv()
        rewards: list[float] = []
        steps = 0
        score = MIN_SCORE
        success = False
        task_error = None
        action_str = "submit(0_spans)"

        log_start(task_id, "pii_redaction_env", model_name)

        try:
            observation = env.reset(seed=42, task_id=task_id)

            predicted_spans = _predict_spans(
                client,
                model_name,
                task_id,
                observation.document_text,
            )
            predicted_spans = _fix_span_offsets(predicted_spans, observation.document_text)

            action = PIIAction(spans=predicted_spans, submit=True)
            action_str = f"submit({len(predicted_spans)}_spans)"
            result = env.step(action)

            steps = env.state.step_count
            step_reward = _clamp(result.reward)
            rewards.append(step_reward)
            task_error = None
            log_step(steps, action_str, step_reward, bool(result.done), task_error)

            score = _clamp(result.final_score) if result.final_score is not None else MIN_SCORE
            success = score >= 0.1
            all_scores.append(score)

        except Exception as task_exc:
            task_error = str(task_exc)[:80]
            print(f"[WARN] task={task_id} failed with: {task_exc}", file=sys.stderr, flush=True)
            rewards = [MIN_SCORE]
            all_scores.append(MIN_SCORE)
            log_step(1, action_str, MIN_SCORE, True, task_error)

        finally:
            try:
                env.close()
            except Exception as close_exc:
                print(f"[WARN] env.close() failed for task={task_id}: {close_exc}", file=sys.stderr, flush=True)
            log_end(success, steps, score, rewards)

    # Final summary
    raw_mean = statistics.mean(all_scores) if all_scores else MIN_SCORE
    clamped_mean = max(MIN_SCORE, min(MAX_SCORE, raw_mean))
    print(json.dumps({"mean_score": clamped_mean, "tasks_completed": len(all_scores)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
