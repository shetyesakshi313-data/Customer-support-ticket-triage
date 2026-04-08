"""
Inference Script — Customer Support Ticket Triage (OpenEnv)
============================================================
MANDATORY env vars:
    API_BASE_URL   LLM endpoint  (default: HuggingFace router)
    MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace API key
    MY_ENV_V4_TASK Task name: easy | medium | hard  (default: easy)

STDOUT FORMAT (strict):
    [START] task=<name> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from my_env_v4 import MyEnvV4Env, MyEnvV4Action, Priority, Department

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "easy")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "support_triage")

MAX_STEPS = 1
TEMPERATURE = 0.0
MAX_TOKENS = 300

# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "easy": textwrap.dedent("""
        You are a customer support triage AI.

        Task:
        Classify the ticket priority using exactly one of these lowercase values:
        low, medium, high, critical

        Guidance:
        - critical: full outage, all users affected, severe revenue loss, major incident
        - high: urgent user-blocking issue, fraud, authentication failure
        - medium: broken functionality, billing discrepancy, shipping delay/problem
        - low: informational request, routine request, minor issue

        Output rules:
        - Return ONLY valid JSON
        - Do not include markdown
        - Do not include explanation
        - Do not include any text before or after the JSON
        - Use lowercase enum values only

        Output format:
        {"priority": "low"}
    """).strip(),

    "medium": textwrap.dedent("""
        You are a customer support triage AI.

        Task:
        Classify the ticket priority and assign the correct department.

        Allowed priority values:
        low, medium, high, critical

        Allowed department values:
        billing, technical, account, shipping, general

        Output rules:
        - Return ONLY valid JSON
        - Do not include markdown
        - Do not include explanation
        - Do not include any text before or after the JSON
        - Use lowercase enum values only

        Output format:
        {"priority": "medium", "department": "billing"}
    """).strip(),

    "hard": textwrap.dedent("""
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
    - Steps must be specific to the reported issue
    - Do not suggest generic app troubleshooting unless the ticket clearly requires it
    - Do not suggest internal engineering actions like checking logs, restarting services, or fixing infrastructure
    - Keep steps short and actionable

    Output rules:
    - Return ONLY valid JSON
    - No explanation
    - No extra text
    - Use lowercase enum values only

    Output format:
    {
      "priority": "high",
      "department": "technical",
      "resolution_steps": ["step 1", "step 2"]
    }
""").strip(),
    }
# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_fallback_action(task: str) -> MyEnvV4Action:
    if task == "easy":
        return MyEnvV4Action(priority=Priority.medium)
    if task == "medium":
        return MyEnvV4Action(
            priority=Priority.medium,
            department=Department.general,
        )
    return MyEnvV4Action(
        priority=Priority.medium,
        department=Department.general,
        resolution_steps=[
            "Review the ticket details",
            "Route to the correct team",
        ],
    )

def clean_json_text(raw: str) -> str:
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    # If extra text is present, try to keep only the outer JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end >= start:
        raw = raw[start:end + 1]

    return raw

def get_model_action(client: OpenAI, ticket_text: str, task: str) -> MyEnvV4Action:
    system_prompt = SYSTEM_PROMPTS[task]
    user_prompt = f'Support ticket:\n"""\n{ticket_text}\n"""'

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        raw = completion.choices[0].message.content or "{}"
        raw = clean_json_text(raw)
        data = json.loads(raw)

        action = MyEnvV4Action(
            priority=Priority(data["priority"]) if "priority" in data else None,
            department=Department(data["department"]) if "department" in data else None,
            resolution_steps=data.get("resolution_steps"),
        )

        # Task-specific validation
        if task == "easy" and action.priority is None:
            return safe_fallback_action(task)

        if task == "medium" and (action.priority is None or action.department is None):
            return safe_fallback_action(task)

        if task == "hard":
            if (
                action.priority is None
                or action.department is None
                or not action.resolution_steps
                or not isinstance(action.resolution_steps, list)
            ):
                return safe_fallback_action(task)

            cleaned_steps = [
                str(step).strip()
                for step in action.resolution_steps
                if str(step).strip()
            ]

            bad_patterns = [
                "internet", "connection", "cache", "update app", "restart app",
                "restart service", "restart server", "server log", "logs", "backend",
                "infrastructure", "database", "gateway service", "connectivity",
                "device model", "os version", "notify customers", "resolution time",
            ]

            filtered_steps = []
            for step in cleaned_steps:
                lower_step = step.lower()
                if not any(pattern in lower_step for pattern in bad_patterns):
                    filtered_steps.append(step)

            if len(filtered_steps) < 2:
                dept = action.department.value if hasattr(action.department, "value") else str(action.department)

                if dept == "billing":
                    filtered_steps = [
                        "verify the charge details in the billing history",
                        "confirm the transaction id and payment date",
                        "initiate a refund or billing correction if the issue is confirmed",
                        "escalate to the billing team with the payment reference",
                    ]
                elif dept == "account":
                    filtered_steps = [
                        "verify the account email and recent access attempt details",
                        "send a password reset or account recovery link",
                        "check whether the account is locked or requires verification",
                        "escalate to the account team if access cannot be restored",
                    ]
                elif dept == "shipping":
                    filtered_steps = [
                        "confirm the order number and delivery address",
                        "check the latest shipment and tracking status",
                        "arrange a replacement or refund if the order is lost or delayed",
                        "share the latest delivery update with the customer",
                    ]
                elif dept == "technical":
                    filtered_steps = [
                        "confirm the exact error and when it occurs",
                        "check whether the issue is reproducible for this account",
                        "guide the user through the affected feature step by step",
                        "escalate to the technical team with the error details if the issue continues",
                    ]
                else:
                    filtered_steps = [
                        "verify the issue details with the customer",
                        "confirm any relevant reference number or recent activity",
                        "provide the next support action based on the reported problem",
                        "escalate to the correct team if additional review is needed",
                    ]

            action.resolution_steps = filtered_steps[:4]

            if len(action.resolution_steps) < 2:
                return safe_fallback_action(task)

        return action

    except Exception:
        return safe_fallback_action(task)

# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    if not API_KEY:
        raise ValueError("HF_TOKEN (or API_KEY) is required")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = MyEnvV4Env(task=TASK_NAME)

    obs = await env.reset()
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    step = 0
    done = False
    rewards: List[float] = []

    try:
        while not done and step < MAX_STEPS:
            step += 1

            action = get_model_action(client, obs.ticket_text, TASK_NAME)
            action_str = json.dumps(action.model_dump(exclude_none=True), ensure_ascii=False)

            obs, reward, done, info = await env.step(action)
            error = info.get("last_action_error")
            rewards.append(float(reward))

            log_step(
                step=step,
                action=action_str,
                reward=float(reward),
                done=bool(done),
                error=error,
            )

    except Exception as exc:
        print(f"[DEBUG] Loop exception: {exc}", flush=True)
        rewards.append(0.0)
        done = True

    finally:
        await env.close()
        score = sum(rewards) / max(len(rewards), 1)
        score = min(max(score, 0.0), 1.0)
        success = len(rewards) > 0 and score > 0.0
        log_end(success=success, steps=len(rewards), score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
