from __future__ import annotations

try:
    from .models import PIIReward, RedactionSpan
except ImportError:
    from models import PIIReward, RedactionSpan


def _base_key(span: RedactionSpan) -> tuple[int, int]:
    return (span.start, span.end)


def _typed_key(span: RedactionSpan) -> tuple[int, int, str]:
    return (span.start, span.end, span.pii_type.value)


def compute_reward(
    predicted: list[RedactionSpan],
    gold: list[RedactionSpan],
    step_count: int,
    terminal_score: float | None = None,
) -> PIIReward:
    predicted_base = {_base_key(span) for span in predicted}
    gold_base = {_base_key(span) for span in gold}
    predicted_typed = {_typed_key(span) for span in predicted}
    gold_typed = {_typed_key(span) for span in gold}

    correct_span_count = len(predicted_base & gold_base)
    correct_type_count = len(predicted_typed & gold_typed)
    false_positive_count = len(predicted_base - gold_base)

    precision = correct_span_count / len(predicted_base) if predicted_base else 0.0
    recall = correct_span_count / len(gold_base) if gold_base else 0.0
    EPS = 1e-6
    if terminal_score is not None:
        total = max(EPS, min(1 - EPS, terminal_score))
    else:
        if precision + recall > 0:
            total = 2 * precision * recall / (precision + recall)
        else:
            total = EPS

    reward = PIIReward(
        span_matches=0.15 * correct_span_count,
        type_matches=0.10 * correct_type_count,
        false_positive_penalty=-0.05 * false_positive_count,
        step_penalty=-0.02 * step_count,
        precision_component=precision,
        recall_component=recall,
        terminal_score=terminal_score if terminal_score is not None else EPS,
    )
    reward.total = max(EPS, min(1 - EPS, total))
    return reward
