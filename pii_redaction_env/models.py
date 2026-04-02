from __future__ import annotations

from enum import Enum
from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PIIType(str, Enum):
    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    PASSPORT = "PASSPORT"
    QUASI_IDENTIFIER = "QUASI_IDENTIFIER"


class RedactionSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)
    pii_type: PIIType
    text: str = Field(..., min_length=1)

    @field_validator("end")
    @classmethod
    def validate_end(cls, value: int, info: Any) -> int:
        start = info.data.get("start")
        if start is not None and value <= start:
            raise ValueError("end must be greater than start")
        return value


class PIIAction(Action):
    spans: list[RedactionSpan] = Field(
        default_factory=list,
        description="Predicted PII spans to redact from the current document.",
    )
    submit: bool = Field(
        default=True,
        description="Whether to finalize the episode after this prediction.",
    )


class PIIReward(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    span_matches: float = Field(default=0.0)
    type_matches: float = Field(default=0.0)
    false_positive_penalty: float = Field(default=0.0)
    step_penalty: float = Field(default=0.0)
    precision_component: float = Field(default=0.0)
    recall_component: float = Field(default=0.0)
    terminal_score: float = Field(default=0.0)
    total: float = Field(default=0.0, ge=-1.0, le=1.0)


class PIIObservation(Observation):
    task_id: str = Field(..., description="Current task identifier.")
    difficulty: str = Field(..., description="Task difficulty label.")
    instructions: str = Field(..., description="Task instructions.")
    document_text: str = Field(..., description="Synthetic document to analyze.")
    predicted_spans: list[RedactionSpan] = Field(default_factory=list)
    reward_details: PIIReward = Field(default_factory=PIIReward)
    final_score: float | None = Field(
        default=None,
        description="Terminal grader score for the current task, when available.",
    )


class PIIState(State):
    task_id: str | None = Field(default=None)
    difficulty: str | None = Field(default=None)
    seed: int | None = Field(default=None, ge=0)
    document_text: str = Field(default="")
    ground_truth_spans: list[RedactionSpan] = Field(default_factory=list)
    predicted_spans: list[RedactionSpan] = Field(default_factory=list)
    reward_history: list[PIIReward] = Field(default_factory=list)
    done: bool = Field(default=False)
    closed: bool = Field(default=False)
