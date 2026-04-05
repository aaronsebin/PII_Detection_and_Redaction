# pii_redaction_env

`pii_redaction_env` is an OpenEnv-compliant environment for seeded synthetic PII detection and redaction. Each episode serves a Faker-generated document at runtime, scores submitted spans deterministically, and exposes the standard `/reset`, `/step`, and `/state` HTTP endpoints through FastAPI on port `7860`.

## Action space

The action is a JSON object with:

- `spans`: a list of `{start, end, pii_type, text}` predictions.
- `submit`: a boolean that finalizes the episode when `true`.

## Observation space

The observation is a JSON object with:

- `task_id`: current task identifier.
- `difficulty`: task difficulty label.
- `instructions`: task-specific redaction goal.
- `document_text`: seeded synthetic document text.
- `predicted_spans`: model predictions stored for the current step.
- `reward_details`: shaped reward breakdown.
- `final_score`: terminal grader score when present.
- `done`: episode termination flag.
- `reward`: scalar environment reward.

## Tasks

| Task | Difficulty | Description | Grader |
| --- | --- | --- | --- |
| `basic_pii_detection` | easy | Detect person, email, and phone spans in a short note. | Exact-match score in `[0, 1]` |
| `mixed_pii_redaction` | medium | Redact mixed direct identifiers including SSN, address, DOB, card, and IP. | Macro F1 in `[0, 1]` |
| `adversarial_quasi_identification` | hard | Redact direct identifiers plus quasi-identifiers that increase re-identification risk. | Weighted F1 with quasi bonus in `[0, 1]` |

## Reward function

Each step uses shaped reward with deterministic components:

- `+0.15` for each predicted span whose boundaries match ground truth.
- `+0.10` for each predicted span whose boundaries and `pii_type` both match ground truth.
- `-0.05` for each false-positive span.
- `-0.02` per step taken.
- On terminal submission, the reward also adds the mean of boundary precision and boundary recall.
- Final reward is clipped to `[-1, 1]`.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn server:app --host 0.0.0.0 --port 7860
```

Run tests:

```bash
pytest
```

## Environment Variables

- `API_BASE_URL`: Base URL for the OpenAI-compatible inference endpoint used by `inference.py`.
- `MODEL_NAME`: Model identifier sent on every LLM request from `inference.py`.
- `HF_TOKEN`: API key passed to the OpenAI client for authenticated requests.

Run the inference script:

```bash
set API_BASE_URL=https://your-openai-compatible-endpoint/v1
set MODEL_NAME=your-model-id
set HF_TOKEN=your_token_here
python inference.py
```

## Inference scores

The inference script evaluates the configured model on all three tasks at `seed=42` and prints structured start, step, and end logs plus the final mean score. Actual values depend on the model snapshot and API access at runtime.

| Model | Seed | basic_pii_detection | mixed_pii_redaction | adversarial_quasi_identification | Mean |
| --- | --- | --- | --- | --- | --- |
| `MODEL_NAME` via `inference.py` | `42` | runtime-evaluated | runtime-evaluated | runtime-evaluated | runtime-evaluated |
