from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from .models import PIIAction, PIIObservation, PIIReward, PIIState
    from .reward import compute_reward
    from .tasks import GRADERS, TASKS, TASK_ORDER
except ImportError:
    from models import PIIAction, PIIObservation, PIIReward, PIIState
    from reward import compute_reward
    from tasks import GRADERS, TASKS, TASK_ORDER


class PIIRedactionEnv(Environment[PIIAction, PIIObservation, PIIState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task_cursor = 0
        self._state = PIIState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> PIIObservation:
        resolved_seed = 42 if seed is None else seed
        requested_task = kwargs.get("task_id")
        task_id = requested_task or TASK_ORDER[self._task_cursor % len(TASK_ORDER)]
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")

        task = TASKS[task_id]
        document_text, gold_spans = task.build_document(seed=resolved_seed)
        self._task_cursor += 1
        self._state = PIIState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task.task_id,
            difficulty=task.difficulty,
            seed=resolved_seed,
            document_text=document_text,
            ground_truth_spans=gold_spans,
            predicted_spans=[],
            reward_history=[],
            done=False,
            closed=False,
        )
        reward = PIIReward()
        EPS = 1e-6
        return PIIObservation(
            done=False,
            reward=EPS,
            task_id=task.task_id,
            difficulty=task.difficulty,
            instructions=task.description,
            document_text=document_text,
            predicted_spans=[],
            reward_details=reward,
            metadata={"task_id": task.task_id, "seed": resolved_seed},
        )

    def step(
        self,
        action: PIIAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> PIIObservation:
        del timeout_s, kwargs
        if self._state.closed:
            raise RuntimeError("Environment is closed — call reset() to start a new episode")
        if self._state.task_id is None:
            raise RuntimeError("Environment must be reset before stepping")
        if self._state.done:
            raise RuntimeError("Episode already completed — call reset() to start a new one")

        self._state.step_count += 1
        self._state.predicted_spans = list(action.spans)

        final_score = None
        if action.submit:
            grader = GRADERS[self._state.task_id]
            final_score = grader(
                self._state.predicted_spans,
                self._state.ground_truth_spans,
            )
            EPS = 1e-6
            final_score = max(EPS, min(1 - EPS, final_score))
            self._state.done = True

        reward = compute_reward(
            predicted=self._state.predicted_spans,
            gold=self._state.ground_truth_spans,
            step_count=self._state.step_count,
            terminal_score=final_score,
        )
        self._state.reward_history.append(reward)

        return PIIObservation(
            done=action.submit,
            reward=reward.total,
            task_id=self._state.task_id,
            difficulty=self._state.difficulty or "",
            instructions=TASKS[self._state.task_id].description,
            document_text=self._state.document_text,
            predicted_spans=list(self._state.predicted_spans),
            reward_details=reward,
            final_score=final_score,
            metadata={
                "task_id": self._state.task_id,
                "seed": self._state.seed,
                "step_count": self._state.step_count,
            },
        )

    @property
    def state(self) -> PIIState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="pii_redaction_env",
            description="Seeded synthetic PII detection and redaction environment with deterministic grading.",
            version="1.0.0",
            author="Codex",
        )

    def close(self) -> None:
        self._state.closed = True
