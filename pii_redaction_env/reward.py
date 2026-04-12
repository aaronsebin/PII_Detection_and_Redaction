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

    gold_span_count = max(1, len(gold_base))
    gold_typed_count = max(1, len(gold_typed))
    penalty_denominator = max(1, len(predicted_base) + len(gold_base))

    span_match_bonus = 0.25 * (correct_span_count / gold_span_count)
    type_match_bonus = 0.20 * (correct_type_count / gold_typed_count)
    false_positive_penalty = -0.10 * (false_positive_count / penalty_denominator)
    # Cap the loop penalty so long episodes do not swamp otherwise useful signal
    step_penalty = -0.02 * min(step_count, 10) / 10.0
    precision_bonus = 0.15 * precision
    recall_bonus = 0.15 * recall

    EPS = 1e-6
    shaped_total = (
        span_match_bonus
        + type_match_bonus
        + false_positive_penalty
        + step_penalty
        + precision_bonus
        + recall_bonus
    )

    if terminal_score is not None:
        total = 0.35 * shaped_total + 0.65 * terminal_score
    else:
        total = shaped_total

    safe_terminal = max(EPS, min(1 - EPS, terminal_score)) if terminal_score is not None else EPS

    reward = PIIReward(
        span_matches=span_match_bonus,
        type_matches=type_match_bonus,
        false_positive_penalty=false_positive_penalty,
        step_penalty=step_penalty,
        precision_component=precision,
        recall_component=recall,
        terminal_score=safe_terminal,
    )
    reward.total = max(EPS, min(1 - EPS, total))
    return reward
