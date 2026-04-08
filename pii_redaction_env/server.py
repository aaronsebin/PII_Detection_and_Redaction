from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

try:
    from .env import PIIRedactionEnv
    from .models import PIIAction, PIIObservation
    from .tasks import TASKS, TASK_ORDER, GRADERS
    from .graders import grade_easy, grade_medium, grade_hard
except ImportError:
    from env import PIIRedactionEnv
    from models import PIIAction, PIIObservation
    from tasks import TASKS, TASK_ORDER, GRADERS
    from graders import grade_easy, grade_medium, grade_hard


app = create_app(
    PIIRedactionEnv,
    PIIAction,
    PIIObservation,
    env_name="pii_redaction_env",
    max_concurrent_envs=4,
)


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata and grader score ranges."""
    task_list = []
    for task_id in TASK_ORDER:
        task = TASKS[task_id]
        task_list.append({
            "task_id": task_id,
            "difficulty": task.difficulty,
            "description": task.description,
            "score_range": {"min": 0.01, "max": 0.99},
        })
    return JSONResponse(content={"tasks": task_list})


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    """Get metadata for a specific task."""
    if task_id not in TASKS:
        return JSONResponse(status_code=404, content={"detail": f"Task '{task_id}' not found"})
    task = TASKS[task_id]
    return JSONResponse(content={
        "task_id": task_id,
        "difficulty": task.difficulty,
        "description": task.description,
        "score_range": {"min": 0.01, "max": 0.99},
    })


@app.post("/tasks/{task_id}/grade")
def grade_task(task_id: str, payload: dict):
    """Grade predicted spans against gold spans for a specific task."""
    if task_id not in GRADERS:
        return JSONResponse(status_code=404, content={"detail": f"Task '{task_id}' not found"})

    try:
        from .models import RedactionSpan
    except ImportError:
        from models import RedactionSpan
    predicted_raw = payload.get("predicted", [])
    gold_raw = payload.get("gold", [])

    try:
        predicted = [RedactionSpan(**s) for s in predicted_raw]
        gold = [RedactionSpan(**s) for s in gold_raw]
    except Exception as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})

    grader = GRADERS[task_id]
    EPS = 1e-6
    score = grader(predicted, gold)
    score = max(EPS, min(1 - EPS, score))

    return JSONResponse(content={"task_id": task_id, "score": score})


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
