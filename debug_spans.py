import os, json
from openai import OpenAI
from env import PIIRedactionEnv
from models import PIIAction, PIIType, RedactionSpan

env = PIIRedactionEnv()
obs = env.reset(seed=42, task_id="basic_pii_detection")

print("Document:")
print(obs.document_text)
print()
print("Gold spans:")
for s in env.state.ground_truth_spans:
    print(f"  [{s.start}:{s.end}] {s.pii_type.value} -> {repr(s.text)}")

client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.getenv("HF_TOKEN")
)

response = client.chat.completions.create(
    model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
    messages=[
        {
            "role": "system",
            "content": "You identify exact PII spans in synthetic documents. Return JSON only with schema: {\"spans\": [{\"start\": int, \"end\": int, \"pii_type\": str, \"text\": str}]}"
        },
        {
            "role": "user",
            "content": "Return all PII spans as exact character offsets.\nDocument:\n" + obs.document_text
        }
    ],
    response_format={"type": "json_object"},
)

print()
print("Model raw output:")
print(response.choices[0].message.content)
print()

# Parse and compare
data = json.loads(response.choices[0].message.content)
print("Predicted spans vs Gold:")
for s in data.get("spans", []):
    text_in_doc = obs.document_text[s["start"]:s["end"]]
    match = "OFFSET MATCH" if text_in_doc == s.get("text") else f"OFFSET MISMATCH (doc has: {repr(text_in_doc)})"
    print(f"  [{s['start']}:{s['end']}] {s['pii_type']} -> {repr(s.get('text'))} | {match}")
