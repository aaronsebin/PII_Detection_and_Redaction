from __future__ import annotations

from dataclasses import dataclass

try:
    from .data.generator import generate_document
    from .graders import grade_easy, grade_hard, grade_medium
except ImportError:
    from data.generator import generate_document
    from graders import grade_easy, grade_hard, grade_medium


@dataclass(frozen=True)
class PIITask:
    task_id: str
    difficulty: str
    description: str
    grader_name: str

    def build_document(self, seed: int) -> tuple[str, list]:
        return generate_document(self.task_id, seed=seed)


TASKS: dict[str, PIITask] = {
    "basic_pii_detection": PIITask(
        task_id="basic_pii_detection",
        difficulty="easy",
        description="Identify straightforward person, email, and phone spans in a short synthetic note.",
        grader_name="grade_easy",
    ),
    "mixed_pii_redaction": PIITask(
        task_id="mixed_pii_redaction",
        difficulty="medium",
        description="Redact mixed direct identifiers across contact, finance, and identity fields.",
        grader_name="grade_medium",
    ),
    "adversarial_quasi_identification": PIITask(
        task_id="adversarial_quasi_identification",
        difficulty="hard",
        description="Catch direct identifiers plus quasi-identifiers that raise re-identification risk.",
        grader_name="grade_hard",
    ),
}

TASK_ORDER = [
    "basic_pii_detection",
    "mixed_pii_redaction",
    "adversarial_quasi_identification",
]

GRADERS = {
    "basic_pii_detection": grade_easy,
    "mixed_pii_redaction": grade_medium,
    "adversarial_quasi_identification": grade_hard,
}
