from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, get_type_hints

from pydantic import ValidationError

from env import PIIRedactionEnv
from graders import grade_easy, grade_hard, grade_medium
from models import PIIAction, PIIObservation, PIIReward, PIIState, PIIType, RedactionSpan
from reward import compute_reward
from tasks import TASK_ORDER, TASKS

ROOT = Path(__file__).resolve().parent


def _report(name: str, passed: bool, detail: str = "") -> bool:
    suffix = f" - {detail}" if detail else ""
    print(f"{'PASS' if passed else 'FAIL'}: {name}{suffix}")
    return passed


def _check_openenv_yaml() -> tuple[bool, str]:
    config_path = ROOT / "openenv.yaml"
    if not config_path.exists():
        return False, "openenv.yaml missing"

    result = subprocess.run(
        ["openenv", "validate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    detail = result.stdout.strip() or result.stderr.strip()
    return result.returncode == 0, detail


def _check_models_importable_and_typed() -> tuple[bool, str]:
    try:
        classes = [PIIAction, PIIObservation, PIIReward, PIIState]
        for cls in classes:
            if not cls.model_fields:
                return False, f"{cls.__name__} has no typed fields"
            if not get_type_hints(cls):
                return False, f"{cls.__name__} missing type hints"
    except Exception as exc:
        return False, str(exc)
    return True, "models loaded with typed fields"


def _sample_span(observation: PIIObservation) -> RedactionSpan:
    text = observation.document_text
    return RedactionSpan(start=0, end=min(4, len(text)), pii_type=PIIType.PERSON, text=text[: min(4, len(text))] or "PII")


def _grader_probe_inputs(env: PIIRedactionEnv) -> tuple[list[RedactionSpan], list[RedactionSpan], list[RedactionSpan]]:
    env.reset(seed=42, task_id="basic_pii_detection")
    easy_gold = list(env.state().ground_truth_spans)
    env.reset(seed=42, task_id="mixed_pii_redaction")
    medium_gold = list(env.state().ground_truth_spans)
    env.reset(seed=42, task_id="adversarial_quasi_identification")
    hard_gold = list(env.state().ground_truth_spans)
    return easy_gold, medium_gold, hard_gold


def _inference_reads_required_env_vars() -> tuple[bool, str]:
    inference_path = ROOT / "inference.py"
    if not inference_path.exists():
        return False, "inference.py missing"

    tree = ast.parse(inference_path.read_text(encoding="utf-8"))
    getenv_args: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "os" and node.func.attr == "getenv":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    getenv_args.add(node.args[0].value)

    required = {"API_BASE_URL", "MODEL_NAME", "HF_TOKEN"}
    missing = sorted(required - getenv_args)
    return not missing, "found all required getenv calls" if not missing else f"missing {', '.join(missing)}"


def _dockerfile_checks() -> tuple[bool, str]:
    dockerfile = ROOT / "Dockerfile"
    if not dockerfile.exists():
        return False, "Dockerfile missing"
    content = dockerfile.read_text(encoding="utf-8")
    required = ["FROM", "EXPOSE 7860", "CMD", 'ENV API_BASE_URL=""', 'ENV MODEL_NAME=""', 'ENV HF_TOKEN=""']
    missing = [item for item in required if item not in content]
    return not missing, "required Dockerfile directives present" if not missing else f"missing {', '.join(missing)}"


def _requirements_checks() -> tuple[bool, str]:
    requirements = ROOT / "requirements.txt"
    if not requirements.exists():
        return False, "requirements.txt missing"
    content = requirements.read_text(encoding="utf-8")
    required = ["openenv-core", "fastapi", "uvicorn", "pydantic", "faker", "openai"]
    missing = [item for item in required if item not in content]
    return not missing, "required dependencies present" if not missing else f"missing {', '.join(missing)}"


def main() -> int:
    checks: list[bool] = []
    env = PIIRedactionEnv()

    try:
        ok, detail = _check_openenv_yaml()
        checks.append(_report("openenv.yaml exists and is valid", ok, detail))

        ok, detail = _check_models_importable_and_typed()
        checks.append(_report("PIIAction, PIIObservation, PIIReward, PIIState models are importable and typed", ok, detail))

        observation = env.reset(seed=42, task_id="basic_pii_detection")
        checks.append(
            _report(
                "env.reset() returns a valid PIIObservation",
                isinstance(observation, PIIObservation),
                observation.task_id if isinstance(observation, PIIObservation) else "",
            )
        )

        redact_result = env.step(PIIAction(spans=[_sample_span(observation)], submit=False))
        try:
            unpacked = redact_result.as_step_result()
            redact_ok = (
                isinstance(unpacked, tuple)
                and len(unpacked) == 4
                and isinstance(unpacked[0], PIIObservation)
                and isinstance(unpacked[1], float)
                and isinstance(unpacked[2], bool)
                and isinstance(unpacked[3], dict)
            )
        except Exception as exc:
            redact_ok = False
            detail = str(exc)
        else:
            detail = "unpacked as (PIIObservation, float, bool, dict)"
        checks.append(_report("env.step() with a redact action returns (PIIObservation, float, bool, dict)", redact_ok, detail))

        submit_result = env.step(PIIAction(spans=env.state().ground_truth_spans, submit=True))
        checks.append(_report("env.step() with submit action returns done=True", submit_result.done is True, f"done={submit_result.done}"))

        try:
            state = env.state()
            state_ok = isinstance(state, PIIState)
            detail = state.task_id or ""
        except Exception as exc:
            state_ok = False
            detail = str(exc)
        checks.append(_report("env.state() returns a valid PIIState", state_ok, detail))

        easy_gold, medium_gold, hard_gold = _grader_probe_inputs(env)
        easy_score = grade_easy(easy_gold, easy_gold)
        medium_score = grade_medium([], medium_gold)
        hard_score = grade_hard(hard_gold[:-1], hard_gold)
        graders_ok = all(isinstance(score, float) and 0.0 < score < 1.0 for score in [easy_score, medium_score, hard_score])
        checks.append(
            _report(
                "grade_easy, grade_medium, grade_hard all return floats strictly between 0.0 and 1.0",
                graders_ok,
                f"easy={easy_score:.6f}, medium={medium_score:.6f}, hard={hard_score:.6f}",
            )
        )

        reward = compute_reward(
            predicted=hard_gold,
            gold=hard_gold,
            step_count=1,
            terminal_score=hard_score,
        )
        reward_ok = isinstance(reward, PIIReward) and -1.0 <= reward.total <= 1.0
        checks.append(
            _report(
                "reward function returns values in [-1.0, 1.0]",
                reward_ok,
                f"total={reward.total:.6f}",
            )
        )

        tasks_ok = (
            len(TASK_ORDER) == 3
            and set(TASK_ORDER) == {"basic_pii_detection", "mixed_pii_redaction", "adversarial_quasi_identification"}
            and all(TASKS[task_id].task_id and TASKS[task_id].difficulty for task_id in TASK_ORDER)
        )
        checks.append(_report("All 3 tasks are enumerable and have valid task_id and difficulty fields", tasks_ok))

        checks.append(_report("inference.py exists in root directory", (ROOT / "inference.py").exists()))

        ok, detail = _dockerfile_checks()
        checks.append(_report("Dockerfile exists and contains FROM, EXPOSE 7860, and CMD", ok, detail))

        ok, detail = _requirements_checks()
        checks.append(_report("requirements.txt exists and contains openenv-core, fastapi, uvicorn, pydantic, faker, openai", ok, detail))

        ok, detail = _inference_reads_required_env_vars()
        checks.append(_report("API_BASE_URL, MODEL_NAME, HF_TOKEN environment variables are read correctly in inference.py", ok, detail))
    except ValidationError as exc:
        checks.append(_report("validator runtime", False, str(exc)))
    finally:
        env.close()

    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
