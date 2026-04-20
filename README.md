# PII Redaction Env

A production-grade OpenEnv environment for training AI agents to detect and redact Personally Identifiable Information (PII) from real-world documents.

Built for the **Meta PyTorch x Hugging Face OpenEnv Hackathon**.

---

## What it does

The agent receives a synthetic document containing PII — names, emails, phone numbers, SSNs, addresses, and more — and must identify and redact every instance while preserving document utility. Tasks range from straightforward detection to adversarial obfuscation and quasi-identifier combinations.

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `basic_pii_detection` | Easy | Detect person, email, and phone spans in a short note |
| `mixed_pii_redaction` | Medium | Redact 8–12 PII instances across 5+ types in an HR document |
| `adversarial_quasi_identification` | Hard | Handle obfuscated PII, quasi-identifiers, and multilingual formats |

---

## Quick Start

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

Run the baseline inference script:

```bash
export HF_TOKEN=your_token_here
python inference.py
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM inference endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | Hugging Face API key | Required |

---

## HuggingFace Space

[Clueless13/pii_redaction_env](https://huggingface.co/spaces/Clueless13/pii_redaction_env)

---

## Tech Stack

- **OpenEnv** — environment spec and deployment
- **FastAPI** — HTTP server
- **Pydantic v2** — typed models
- **Faker** — synthetic document generation
- **Docker** — containerized deployment
