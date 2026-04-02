from __future__ import annotations

from pii_redaction_env.env import PIIRedactionEnv
from pii_redaction_env.graders import grade_easy, grade_hard, grade_medium
from pii_redaction_env.models import PIIAction, PIIObservation


def test_reset_returns_valid_observation() -> None:
    env = PIIRedactionEnv()
    observation = env.reset(seed=42, task_id="basic_pii_detection")
    assert isinstance(observation, PIIObservation)
    assert observation.task_id == "basic_pii_detection"
    assert observation.document_text
    assert observation.done is False


def test_step_submit_returns_done_true() -> None:
    env = PIIRedactionEnv()
    env.reset(seed=42, task_id="basic_pii_detection")
    result = env.step(PIIAction(spans=[], submit=True))
    assert result.done is True


def test_all_graders_return_unit_interval_float() -> None:
    env = PIIRedactionEnv()
    env.reset(seed=42, task_id="basic_pii_detection")
    easy_gold = env.state.ground_truth_spans
    env.reset(seed=42, task_id="mixed_pii_redaction")
    medium_gold = env.state.ground_truth_spans
    env.reset(seed=42, task_id="adversarial_quasi_identification")
    hard_gold = env.state.ground_truth_spans

    easy_score = grade_easy(easy_gold, easy_gold)
    medium_score = grade_medium([], medium_gold)
    hard_score = grade_hard(hard_gold, hard_gold)

    assert isinstance(easy_score, float)
    assert isinstance(medium_score, float)
    assert isinstance(hard_score, float)
    assert 0.0 <= easy_score <= 1.0
    assert 0.0 <= medium_score <= 1.0
    assert 0.0 <= hard_score <= 1.0
