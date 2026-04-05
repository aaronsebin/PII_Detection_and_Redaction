from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

try:
    from .models import PIIType, RedactionSpan
except ImportError:
    from models import PIIType, RedactionSpan

_EPSILON = 1e-6


def _span_key(span: RedactionSpan) -> tuple[int, int, str]:
    return (span.start, span.end, span.pii_type.value)


def _group_by_type(
    spans: Iterable[RedactionSpan],
) -> dict[PIIType, set[tuple[int, int, str]]]:
    grouped: dict[PIIType, set[tuple[int, int, str]]] = defaultdict(set)
    for span in spans:
        grouped[span.pii_type].add(_span_key(span))
    return dict(grouped)


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0 - _EPSILON
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return _EPSILON
    return max(
        _EPSILON,
        min(1.0 - _EPSILON, 2 * precision * recall / (precision + recall)),
    )


def grade_easy(predicted: list[RedactionSpan], gold: list[RedactionSpan]) -> float:
    return (
        1.0 - _EPSILON
        if {_span_key(span) for span in predicted}
        == {_span_key(span) for span in gold}
        else _EPSILON
    )


def grade_medium(predicted: list[RedactionSpan], gold: list[RedactionSpan]) -> float:
    predicted_by_type = _group_by_type(predicted)
    gold_by_type = _group_by_type(gold)
    labels = sorted(set(predicted_by_type) | set(gold_by_type), key=lambda item: item.value)
    if not labels:
        return 1.0 - _EPSILON

    scores: list[float] = []
    for label in labels:
        pred_set = predicted_by_type.get(label, set())
        gold_set = gold_by_type.get(label, set())
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        scores.append(_safe_f1(tp, fp, fn))
    return max(_EPSILON, min(1.0 - _EPSILON, sum(scores) / len(scores)))


def grade_hard(predicted: list[RedactionSpan], gold: list[RedactionSpan]) -> float:
    predicted_by_type = _group_by_type(predicted)
    gold_by_type = _group_by_type(gold)
    labels = sorted(set(predicted_by_type) | set(gold_by_type), key=lambda item: item.value)
    if not labels:
        return 1.0 - _EPSILON

    total_weight = 0.0
    weighted_score = 0.0
    quasi_recall = 0.0

    for label in labels:
        pred_set = predicted_by_type.get(label, set())
        gold_set = gold_by_type.get(label, set())
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        f1 = _safe_f1(tp, fp, fn)
        weight = 1.5 if label == PIIType.QUASI_IDENTIFIER else 1.0
        weighted_score += weight * f1
        total_weight += weight
        if label == PIIType.QUASI_IDENTIFIER:
            quasi_recall = tp / (tp + fn) if tp + fn else 1.0

    base = weighted_score / total_weight if total_weight else 0.0
    bonus = 0.1 * quasi_recall
    return max(_EPSILON, min(1.0 - _EPSILON, base + bonus))
