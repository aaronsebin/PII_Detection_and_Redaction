"""Test the HF Space API endpoints like the Phase 2 validator would."""
import requests
import json

BASE = "https://clueless13-pii-redaction-env.hf.space"

tasks = ["basic_pii_detection", "mixed_pii_redaction", "adversarial_quasi_identification"]

for task_id in tasks:
    print(f"\n=== Testing {task_id} ===")
    
    # Reset
    r = requests.post(f"{BASE}/reset", json={"seed": 42, "task_id": task_id})
    reset_data = r.json()
    reset_reward = reset_data.get("reward")
    reset_done = reset_data.get("done")
    obs = reset_data.get("observation", {})
    final_score_reset = obs.get("final_score")
    reward_total_reset = obs.get("reward_details", {}).get("total")
    print(f"  RESET: reward={reset_reward}, done={reset_done}, final_score={final_score_reset}, reward_details.total={reward_total_reset}")
    
    # Step with empty spans (worst case)
    r2 = requests.post(f"{BASE}/step", json={"action": {"spans": [], "submit": True}})
    if r2.status_code != 200:
        print(f"  STEP FAILED: {r2.status_code} {r2.text[:200]}")
        continue
    step_data = r2.json()
    step_reward = step_data.get("reward")
    step_done = step_data.get("done")
    obs2 = step_data.get("observation", {})
    final_score_step = obs2.get("final_score")
    reward_total_step = obs2.get("reward_details", {}).get("total")
    print(f"  STEP:  reward={step_reward}, done={step_done}, final_score={final_score_step}, reward_details.total={reward_total_step}")
    
    # Check for exact 0.0 or 1.0
    for name, val in [
        ("reset.reward", reset_reward),
        ("reset.final_score", final_score_reset),
        ("reset.reward_details.total", reward_total_reset),
        ("step.reward", step_reward),
        ("step.final_score", final_score_step),
        ("step.reward_details.total", reward_total_step),
    ]:
        if val is not None and (val == 0.0 or val == 1.0):
            print(f"  *** PROBLEM: {name} = {val} (exactly 0.0 or 1.0) ***")
