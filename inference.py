"""
Inference Script — Customer Support Ticket Triage
"""

import json
import os
import sys
from typing import List, Dict, Any

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "easy")

# Point this to your live HF Space
ENV_BASE_URL = os.getenv(
    "MY_ENV_V4_SPACE_URL",
    "https://sakshi313-support-triage-env.hf.space"
).rstrip("/")

MAX_STEPS = 1
TEMPERATURE = 0.0
MAX_TOKENS = 300


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPTS = {
    "easy": """
You are a customer support triage AI.

Task:
Classify the ticket priority using exactly one of these lowercase values:
low, medium, high, critical

Guidance:
- critical: full outage, all users affected, severe revenue loss, major incident
- high: urgent user-blocking issue, fraud, authentication failure
- medium: broken functionality, billing discrepancy, shipping delay/problem
- low: informational request, routine request, minor issue

Return ONLY valid JSON in this format:
{"priority": "low"}
""".strip(),

    "medium": """
You are a customer support triage AI.

Task:
Classify the ticket priority and assign the correct department.

Allowed priority values:
low, medium, high, critical

Allowed department values:
billing, technical, account, shipping, general

Return ONLY valid JSON in this format:
{"priority": "medium", "department": "billing"}
""".strip(),

    "hard": """
You are a senior customer support triage AI.

Task:
1. Classify ticket priority
2. Assign the correct department
3. Provide 2 to 4 customer-support resolution steps

Allowed priority values:
low, medium, high, critical

Allowed department values:
billing, technical, account, shipping, general

Resolution steps rules:
- Steps must be from a support agent perspective
- Steps must be specific to the issue
- Do not suggest engineering/internal actions like checking logs or restarting servers
- Keep steps short and actionable

Return ONLY valid JSON in this format:
{
  "priority": "high",
  "department": "technical",
  "resolution_steps": ["step 1", "step 2"]
}
""".strip(),
}


def clean_json_text(raw: str) -> str:
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end >= start:
        raw = raw[start:end + 1]
    return raw


def safe_fallback_action(task: str) -> Dict[str, Any]:
    if task == "easy":
        return {"priority": "medium"}
    if task == "medium":
        return {"priority": "medium", "department": "general"}
    return {
        "priority": "medium",
        "department": "general",
        "resolution_steps": [
            "Review the ticket details",
            "Confirm the issue with the customer",
        ],
    }


def filter_steps(department: str, steps: List[str]) -> List[str]:
    bad_patterns = [
        "internet", "connection", "cache", "restart service", "restart server",
        "logs", "backend", "database", "infrastructure", "device model", "os version"
    ]

    cleaned = [str(s).strip() for s in steps if str(s).strip()]
    filtered = [
        s for s in cleaned
        if not any(p in s.lower() for p in bad_patterns)
    ]

    if len(filtered) >= 2:
        return filtered[:4]

    if department == "billing":
        return [
            "Verify the charge details in the billing history",
            "Confirm the transaction ID and payment date",
            "Initiate a refund or billing correction if confirmed",
        ]
    if department == "account":
        return [
            "Verify the account email and recent access attempt details",
            "Send a password reset or recovery link",
            "Escalate to the account team if access is not restored",
        ]
    if department == "shipping":
        return [
            "Confirm the order number and delivery address",
            "Check the latest shipment and tracking status",
            "Arrange a replacement or refund if the order is delayed or lost",
        ]
    if department == "technical":
        return [
            "Confirm the exact error and when it occurs",
            "Check whether the issue can be reproduced for this account",
            "Escalate to the technical team with the error details if needed",
        ]

    return [
        "Verify the issue details with the customer",
        "Confirm any related reference number or recent activity",
        "Escalate to the correct team if further review is needed",
    ]


def get_model_action(client: OpenAI, ticket_text: str, task: str) -> Dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user", "content": f'Support ticket:\n"""\n{ticket_text}\n"""'},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        raw = completion.choices[0].message.content or "{}"
        raw = clean_json_text(raw)
        data = json.loads(raw)

        if task == "easy":
            priority = data.get("priority")
            if priority not in {"low", "medium", "high", "critical"}:
                return safe_fallback_action(task)
            return {"priority": priority}

        if task == "medium":
            priority = data.get("priority")
            department = data.get("department")
            if priority not in {"low", "medium", "high", "critical"}:
                return safe_fallback_action(task)
            if department not in {"billing", "technical", "account", "shipping", "general"}:
                return safe_fallback_action(task)
            return {"priority": priority, "department": department}

        priority = data.get("priority")
        department = data.get("department")
        steps = data.get("resolution_steps", [])

        if priority not in {"low", "medium", "high", "critical"}:
            return safe_fallback_action(task)
        if department not in {"billing", "technical", "account", "shipping", "general"}:
            return safe_fallback_action(task)
        if not isinstance(steps, list):
            steps = []

        steps = filter_steps(department, steps)
        if len(steps) < 2:
            return safe_fallback_action(task)

        return {
            "priority": priority,
            "department": department,
            "resolution_steps": steps[:4],
        }

    except Exception:
        return safe_fallback_action(task)


def post_json(url: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        response = requests.post(url, json=payload or {}, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise RuntimeError(f"HTTP call failed for {url}: {e}")


def main() -> None:
    rewards: List[float] = []

    try:
        if not HF_TOKEN:
            raise ValueError("HF_TOKEN is required")

        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

        log_start(task=TASK_NAME, env="support_triage", model=MODEL_NAME)

        obs = post_json(f"{ENV_BASE_URL}/reset")
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            step += 1
            ticket_text = obs.get("ticket_text", "")
            action = get_model_action(client, ticket_text, TASK_NAME)

            result = post_json(f"{ENV_BASE_URL}/step", action)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", True))
            info = result.get("info", {}) or {}
            error = info.get("last_action_error")

            rewards.append(reward)
            log_step(
                step=step,
                action=json.dumps(action, ensure_ascii=False),
                reward=reward,
                done=done,
                error=error,
            )

            obs = result.get("observation", {})

    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        rewards.append(0.0)

    score = sum(rewards) / max(len(rewards), 1)
    score = min(max(score, 0.0), 1.0)
    success = len(rewards) > 0 and score > 0.0
    log_end(success=success, steps=len(rewards), score=score, rewards=rewards)


if __name__ == "__main__":
    main()